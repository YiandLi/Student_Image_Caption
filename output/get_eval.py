import json

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
import matplotlib.pyplot as plt
import pandas as pd


def coco_caption_eval(references, hypotheses):
    """
    references = {
        "image1": ["there is a cat on the mat.", "a cat is on the mat."],
        # "image2": ["A dog is running in the park.", "Children are playing in the playground."]
    }
    hypotheses = {
        "image1": ["the cat is on the mat."],
        # "image2": ["A dog is playing in the field."]
    }
    bleu_scorer = Bleu(n=4)
    bleu_scores, _ = bleu_scorer.compute_score(references, hypotheses)
    """
    bleu4_scores, _ = Bleu(n=4).compute_score(references, hypotheses)
    bleu4_scores = bleu4_scores[-1]
    cider_score, _ = Cider().compute_score(references, hypotheses)
    
    return {
        "bleu4": bleu4_scores,
        "cider_score": cider_score
    }


if __name__ == '__main__':
    log_reader = open('log.txt', "r")
    res_reader = open('result.txt', 'r')
    
    bleu3, bleu4, rouge, cider, ep = [], [], [], [], []
    epoch = 0
    for log, rees in zip(log_reader, res_reader):
        print('epoch', epoch)
        ep.append(epoch)
        log = json.loads(log)
        rees = json.loads(rees)
        val_result = list(rees.values())[0]['val_result']
        val_ground_true = list(rees.values())[0]['val_ground_true']
        a = coco_caption_eval(val_ground_true, val_result)
        
        bleu3.append(log['val_bleu3'])
        rouge.append(log['val_rouge'])
        bleu4.append(a['bleu4'])
        cider.append(a['cider_score'])
        epoch += 1
    
    plt.figure(figsize=(10, 6))
    plt.plot(ep, bleu3, label='BLEU-3', marker='o')
    plt.plot(ep, bleu4, label='BLEU-4', marker='s')
    plt.plot(ep, rouge, label='ROUGE', marker='^')
    plt.plot(ep, cider, label='CIDEr', marker='D')
    
    # Annotate maximum values
    plt.text(ep[bleu3.index(max(bleu3))], max(bleu3), f'Max: {max(bleu3):.2f}', ha='right', va='bottom')
    plt.text(ep[bleu4.index(max(bleu4))], max(bleu4), f'Max: {max(bleu4):.2f}', ha='right', va='bottom')
    plt.text(ep[rouge.index(max(rouge))], max(rouge), f'Max: {max(rouge):.2f}', ha='right', va='bottom')
    plt.text(ep[cider.index(max(cider))], max(cider), f'Max: {max(cider):.2f}', ha='right', va='bottom')
    
    # Add title and labels
    plt.title('Evaluation Metrics Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    
    # Set x-axis range from 0 to 50
    plt.xlim(0, 50)
    
    # Add legend
    plt.legend()
    
    # Show plot
    plt.grid(True)
    plt.show()
    
    data = {
        'Metric': ['BLEU-3', 'BLEU-4', 'ROUGE', 'CIDEr'],
        'Min': [min(bleu3), min(bleu4), min(rouge), min(cider)],
        'Max': [max(bleu3), max(bleu4), max(rouge), max(cider)]
    }
    
    df = pd.DataFrame(data)
    # df = df.to_latex(index=False)
    print(df)
