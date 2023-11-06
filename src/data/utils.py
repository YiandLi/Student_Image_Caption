import re
import json
import os

from pycocoevalcap.cider.cider import Cider

import utils
import torch.distributed as dist
from pycocoevalcap.spice.spice import Spice


def pre_caption(caption, max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')
    
    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])
    
    return caption


def pre_question(question, max_ques_words=50):
    question = re.sub(
        r"([.!\"()*#:;~])",
        '',
        question.lower(),
    )
    question = question.rstrip(' ')
    
    # truncate question
    question_words = question.split(' ')
    if len(question_words) > max_ques_words:
        question = ' '.join(question_words[:max_ques_words])
    
    return question


def save_result(result, result_dir, filename, remove_duplicate=''):
    result_file = os.path.join(result_dir, '%s_rank%d.json' % (filename, utils.get_rank()))
    final_result_file = os.path.join(result_dir, '%s.json' % filename)
    
    json.dump(result, open(result_file, 'w'))
    
    dist.barrier()
    
    if utils.is_main_process():
        # combine results from all processes
        result = []
        
        for rank in range(utils.get_world_size()):
            result_file = os.path.join(result_dir, '%s_rank%d.json' % (filename, rank))
            res = json.load(open(result_file, 'r'))
            result += res
        
        if remove_duplicate:
            result_new = []
            id_list = []
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new
        
        json.dump(result, open(final_result_file, 'w'))
        print('result file saved to %s' % final_result_file)
    
    return final_result_file


from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge


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
    bleu3_scores, _ = Bleu(n=3).compute_score(references, hypotheses)
    bleu3_scores = bleu3_scores[-1]
    rouge_score, _ = Rouge().compute_score(references, hypotheses)
    spice_score, _ = Spice().compute_score(references, hypotheses)
    cider_score, _ = Cider().compute_score(references, hypotheses)
    
    return {
        "bleu3": bleu3_scores,
        "rouge": rouge_score,
        "spice": spice_score,
        "cider": cider_score
    }
