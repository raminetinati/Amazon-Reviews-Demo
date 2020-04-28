from transformers import DistilBertTokenizer, BertTokenizer, DistilBertModel, BertModel
from multiprocessing import  Pool
import multiprocessing as mp
import torch.multiprocessing as torchmp
from functools import partial
import tensorflow as tf
from transformers import (
    BertConfig, 
    BertTokenizer,
    TFBertForSequenceClassification
)

from collections import namedtuple
from typing import List, Tuple


def bert_load_components(global_vars):
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    global_vars['bert_tokenizer'] = tokenizer
    global_vars['bert_model'] = model

    return global_vars




def funct_token(df):

        df['bert_tokens']= df['processed_text'].apply(
            (lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=512))
        )
        return df
    
def bert_tokenize_and_pad(global_vars, df, text_col):
        
    #first we need to reconstruct our pre_processed reviews to a new column which is a string.
    
    col_to_add = 'processed_text'
    tmp = df[text_col]
    xs = []
    for entry in tmp:
        res = str(entry).strip('][').split(', ') 
        res = ' '.join(res)
        xs.append(res)
        
    df['processed_text'] = xs   
    print('Reprocessed text as Strings, New column {}'.format(col_to_add))
    
    tokenizer = global_vars['bert_tokenizer'] 

    cores = mp.cpu_count()
    print('Enabling parallel processing. {} Core Available'.format(cores))

    df_split = np.array_split(df, cores)
    pool = Pool(cores)
         
    pool_results = pool.map(funct_token, df_split)
    pool.close()
    pool.join()
    
    parts = pd.concat(pool_results, axis=0)
    toks = parts['bert_tokens']
    
    print('Total Strings Parsed {} Core Available'.format(len(toks)))

#     with mp.Pool(mp.cpu_count()) as pool:
#         toks = pool.map(func, df['processed_text'])

    
#     toks = ddata['processed_text'].map_partitions(lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=512)).compute(get=get)  

#     toks = df['processed_text'].apply(
#         (lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=512)
#         )
#     )
    
    #find max list length and pad to that length
    max_len = 0
    for i in toks.values:
        if len(i) > max_len:
            max_len = len(i)

    print('Max Sequence Length {}'.format(max_len))
    
    padded_toks = np.array([i + [0]*(max_len-len(i)) for i in toks.values]) 
       
    print('Padded Dimensions {}'.format(np.array(padded_toks).shape))
    
    mask = np.where(padded_toks != 0, 1, 0)
    
    return padded_toks, mask
  
bert_padded_toks, bert_mask = bert_tokenize_and_pad(global_vars, sampled_data, 'review_body')



def bert_save_load_object(obj_to_save, path = 'data', file_name_prefix='', operation='save', chunkSize = 100000):
    
    
    loaded = []
    #first split the df as it's too big probably
    listOfobs= list()
    if operation == 'save':

        numberChunks = len(obj_to_save) // chunkSize + 1
        for i in range(numberChunks):
            listOfobs.append(obj_to_save[i*chunkSize:(i+1)*chunkSize])
            
        for i in range(0, len(listOfobs)):
            chunk_df = listOfobs[i]
            obj_tmp_name_prefix = '{}/{}_part_{}.pkl'.format(path, file_name_prefix, str(i))
            pickle.dump(chunk_df, open(obj_tmp_name_prefix,'wb'))
                       
        return obj_to_save
                       
    if operation == 'load':
        root_name = '{}/{}_*.pkl'.format(path, file_name_prefix)
        files = glob.glob(root_name)
        for fl in files:       
            print(fl)
            obj = pickle.load( open(filename, "rb" ) )
            loaded.append(obj)
                       
        return list(itertools.chain.from_iterable(loaded))
    
    






# torchmp.set_start_method('fork')

def bert_eval_embeddings(model, i, input_ids_batch, attention_mask_batch):
    

    with torch.no_grad():
        print('Computing Hidden States for Batch Length {}'.format(len(input_ids_batch)))
        last_hidden_states = model(input_ids_batch, attention_mask=attention_mask_batch)
        return_dict = {}
        return_dict[i] = last_hidden_states
#         queue.put(return_dict)
        print('Batch Saved In Return_Dict Index {}'.format(i))
        return last_hidden_states


def bert_process_embeddings(global_vars, padded, mask,):
#     torchmp.set_start_method('spawn')

#     print(model.config)
    model = BertModel.from_pretrained('bert-base-uncased')
    model.share_memory()
    num_of_workers =  4

    tmp = []
    batch_size = 4096
    batches = len(padded) // batch_size
    results = []
    print('Batch Size {}. Total Batches {}'.format(batch_size, batches))
    for i in range(0, batches):

        lower = batch_size * i
        upper = batch_size * (i+1)
        if i == batches:
            upper = len(padded)
        print('Batch {} : {}'.format(lower,upper))

        input_ids = torch.tensor(padded[lower:upper])  
        attention_mask = torch.tensor(mask[lower:upper])
        print('Input length {}, Attention_mask length {}'.format(len(input_ids), len(attention_mask)))

        input_ids_split = torch.chunk(input_ids, num_of_workers)
        attention_mask_batch_split = torch.chunk(attention_mask, num_of_workers)

        print('Splits {}'.format(len(attention_mask_batch_split)))
        queue = torchmp.SimpleQueue()

        results_batch = []
        for i in range(0, len(input_ids_split)):
            print('Pool {}'.format(i))
            results_batch.append(bert_eval_embeddings(model, i, input_ids_split[i],attention_mask_batch_split[i]))
        results_batch = list(itertools.chain.from_iterable(results_batch))
        results.append(results_batch)


    last_hidden_states = list(itertools.chain.from_iterable(result_arrays))
    
    features = last_hidden_states[0][:,0,:].numpy()

    return features, input_ids, attention_mask





if __name__ == "__main__":
    
    
    
    global_vars = bert_load_components({})
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    bert_padded_toks = bert_save_load_object(bert_padded_toks, path='data', file_name_prefix = 'bert_padded_toks.pkl', operation='save')
    
    bert_mask = bert_save_load_object(bert_mask, path='data', file_name_prefix = 'bert_mask.pkl', operation='save')

    
    features2, input_ids2, attention_mask2 = bert_process_embeddings(global_vars, bert_padded_toks, bert_mask)


    
