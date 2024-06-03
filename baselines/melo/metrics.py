import torch
import logging

from baselines.melo.utils import *
LOG = logging.getLogger(__name__)

# For zsRE and F-NLI
def retain_rate(pre_logits, post_logits, mask=None):
    if pre_logits.shape[-1] == 1:
        pre_logits = pre_logits.squeeze(-1)
    if post_logits.shape[-1] == 1:
        post_logits = post_logits.squeeze(-1)

    assert pre_logits.shape == post_logits.shape
    assert pre_logits.shape[0] == mask.shape[0]

    if pre_logits.dim() == 1:
        # binary classification
        pre_preds = pre_logits > 0
        post_preds = post_logits > 0
        retain = (pre_preds == post_preds).float().mean()
    elif pre_logits.dim() == 3:
        # sequence modeling
        pre_preds = pre_logits.argmax(-1)
        post_preds = post_logits.argmax(-1)
        match = (pre_preds == post_preds) * mask
        retain = (match.sum(-1) == mask.sum(-1)).float().mean()
    else:
        raise NotImplementedError

    return retain.item()


def is_acc_error(model, tokens):
    # Check whether or not the model's prediction for a batch element is correct
    labels = tokens["labels"]
    logits = model(**tokens).logits
    probs = torch.softmax(logits, -1).squeeze()
    argmaxs = torch.argmax(probs, dim=-1).squeeze()
    return labels != argmaxs


def Accuracy(alg, tokens):
    labels = tokens["labels"]
    new_tokens = {f"{k}": v for k, v in tokens.items() if k != "labels"}
    logits = alg.model(**new_tokens).logits
    probs = torch.softmax(logits, -1).squeeze()
    argmaxs = torch.argmax(probs, dim=-1).squeeze()
    return (labels == argmaxs).float().mean()


def is_qa_error(model, tokens):
    preds = model.generate(tokens["input_ids"], max_length=50).squeeze()  # Run model to get its predictions
    labels = tokens["labels"]  # [tokens["labels"] != -100]

    if (len(preds) != len(labels)) or ((preds == labels).sum() != len(preds)):
        return True
    else:
        return False


def PPL(alg, batch):
    input_ids = batch["input_ids"][:, :1024]  # .to(device)
    if "labels" not in batch:
        batch["labels"] = batch["input_ids"][:, :1024].clone()
    else:
        batch["labels"] = batch["labels"][:, :1024].clone()

    with torch.no_grad():
        #outputs = alg.model.model(input_ids=input_ids, labels=target_ids)
        outputs = alg.model(**batch)
        nll = outputs.loss

    ppl = torch.exp(nll)  # .clip(0, 100)
    return ppl



def F1_ACC(alg, batch):
    try:
        preds = alg.generate(batch["input_ids"], max_length=50, pad_token_id = tokenizer.eos_token_id,).squeeze()
        f1 = F1(preds, batch, alg.model_tok)
        acc = ACC(preds, batch, alg.model_tok)
        return f1, acc
    except Exception as e:
        raise e

def F1(preds, batch, tok):
    try:
        f1_list = []
        for p, g in zip(preds,batch["labels"]):
            p = p[p !=  tok.pad_token_id].cpu().squeeze()
            g = g[g != -100].cpu().squeeze()  # -100 might be nonsense
            num_same = len(np.intersect1d(p, g))
            len_pred = len(p)
            len_gold = len(g)
            precision = num_same / len_pred
            recall = 1.0 * num_same / len_gold
            f1 = (2 * precision * recall) / (precision + recall)
            f1_list.append(f1)
    except:
        return 0.

    return sum(f1_list) / len(f1_list)


def ACC(preds, batch, tok):
    decode_preds = tok.batch_decode(preds,skip_special_tokens=True)
    gold_labels = batch['labels']
    gold_labels = gold_labels.masked_fill(gold_labels == -100,tok.pad_token_id)
    decode_labels = tok.batch_decode(gold_labels,skip_special_tokens=True)
    assert len(decode_labels) == len(decode_preds), "Lengths of decode_preds and decode_labels should be the same"
    count = 0.
    for pred,label in zip(decode_preds, decode_labels):
        if pred == label:
            count = count + 1
    return count/len(decode_preds)

def gen_ACC(alg, questions, answers):
    tok = alg.model_tok
    ES_acc = []
    for question, answer in zip(questions, answers):
        ans_tok = tok.encode(answer)

        input_prompt = [question + tok.decode(ans_tok[:i]) for i in range(len(ans_tok))]
        input_target = [tok.decode(ans_tok[i]) for i in range(len(ans_tok))]
        
        prompt_tok = tok(
            input_prompt,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            logits = alg.model(**prompt_tok).logits
            last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
            to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
            gathered = torch.gather(logits, 1, to_gather).squeeze(1)
            ans = torch.argmax(gathered, dim=1)

            correct_id = tok(input_target, padding=True, return_tensors="pt").to("cuda")[
                "input_ids"
            ]
            # Temporary hack to deal with foreign characters.
            correct_id = correct_id[:, 0].squeeze()

            ES_acc += [(ans == correct_id).detach().cpu().numpy().mean()]
    return sum(ES_acc) / len(ES_acc)