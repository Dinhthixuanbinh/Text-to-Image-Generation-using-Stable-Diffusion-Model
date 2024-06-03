import torch

def get_text_representation(text, text_tokenizer, text_model , 
                            device, truncation = True, 
                            padding = 'max_length', max_length = 77):
    token_output = text_tokenizer(text, 
                                  truncation = truncation,
                                  padding = padding,
                                  return_attention_mask = True,
                                  max_length = max_length
                                  )
    indexed_tokens = token_output['input_ids']
    att_masks = token_output['attention_mask']
    tokens_tensor = torch.tensor(indexed_tokens).to(device)
    mask_tensor = torch.tensor(att_masks).to(device)
    text_embed = text_model(tokens_tensor, attention_mask = mask_tensor).last_hidden_state    
    return text_embed

def drop_text_condition(text_embed, im, empty_text_embed, text_drop_prob) :
    if text_drop_prob > 0:
        text_prob_mask = torch.zeros((im.shape[0]), device = im.device).float().uniform_
        (0,
         1) < text_drop_prob
        assert empty_text_embed is not None, ('Text Conditioning required as well as text dropping but empty text representation not created')
        text_embed[text_prob_mask, :, :] = empty_text_embed[0]
    return text_embed
   

