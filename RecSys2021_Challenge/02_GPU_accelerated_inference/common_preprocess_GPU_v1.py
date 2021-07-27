# Copyright 2021 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
# coding: utf-8

############################################################################
# THIS FILE PROCESSES TSV DATA, ENGINEERS FEATURES, AND SAVES CONVERTED PARQUET
############################################################################

# LOAD GPU LIBRARIES, USE 1 GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import cudf, cupy as cp, gc
print('cudf version',cudf.__version__)


# TSV COLUMN NAMES
features = [
    'text_tokens',    ###############
    'hashtags',       #Tweet Features
    'tweet_id',       #
    'media',          #
    'links',          #
    'domains',        #
    'tweet_type',     #
    'language',       #
    'timestamp',      ###############
    'a_user_id',              ###########################
    'a_follower_count',       #Engaged With User Features
    'a_following_count',      #
    'a_is_verified',          #
    'a_account_creation',     ###########################
    'b_user_id',              #######################
    'b_follower_count',       #Engaging User Features
    'b_following_count',      #
    'b_is_verified',          #
    'b_account_creation',     #######################
    'b_follows_a',    #################### Engagement Features
    'reply',          #Target Reply
    'retweet',        #Target Retweet    
    'retweet_comment',#Target Retweet with comment
    'like',           #Target Like
                      ####################
]
features_submission = features[:20]

# LABEL ENCODING MAP FOR MEDIA COLUMN
MAP_MEDIA = {
 '': 0,
 'Photo': 1,
 'Photo\tPho': 2,
 'Video': 3,
 'GIF': 4,
 'Video\tVid': 5,
 'Photo\tVid': 6,
 'Video\tPho': 7,
 'GIF\tPhoto': 8,
 'Photo\tGIF': 9,
 'GIF\tGIF': 10,
 'GIF\tVideo': 11,
 'Video\tGIF': 12,
 'GIF\tGIF\tG': 13
}
MAP_MEDIA = cudf.DataFrame( cudf.Series(MAP_MEDIA) ).reset_index()
MAP_MEDIA.columns = ['media2','media']
MAP_MEDIA['media'] = MAP_MEDIA['media'].astype('int8')

# LABEL ENCODING MAP FOR TWEET TYPE COLUMN
MAP_TYPE = {'':0, 'TopLevel': 0, 'Retweet': 1, 'Quote': 2}
MAP_TYPE = cudf.DataFrame( cudf.Series(MAP_TYPE) ).reset_index()
MAP_TYPE.columns = ['tweet_type2','tweet_type']
MAP_TYPE['tweet_type'] = MAP_TYPE['tweet_type'].astype('int8')

# LABEL ENCODING MAP FOR LANGUAGE COLUMN
MAP_LANG = {
 '': 0,
 '488B32D24BD4BB44172EB981C1BCA6FA': 0,
 'E7F038DE3EAD397AEC9193686C911677': 1,
 'B0FA488F2911701DD8EC5B1EA5E322D8': 2,
 'B8B04128918BBF54E2E178BFF1ABA833': 3,
 '313ECD3A1E5BB07406E4249475C2D6D6': 4,
 '1F73BB863A39DB62B4A55B7E558DB1E8': 5,
 '9FCF19233EAD65EA6E32C2E6DC03A444': 6,
 '9A78FC330083E72BE0DD1EA92656F3B5': 7,
 '8729EBF694C3DAF61208A209C2A542C8': 8,
 'E6936751CBF4F921F7DE1AEF33A16ED0': 9,
 '7F4FAB1EB12CD95EDCD9DB2A6634EFCE': 10,
 'B4DC2F82961F1263E90DF7A942CCE0B2': 11,
 '310ECD7D1E42216E3C1B31EFDDFC72A7': 12,
 '5A0759FB938B1D9B1E08B7A3A14F1042': 13,
 '2F548E5BE0D7F678E72DDE31DFBEF8E7': 14,
 '5B6973BEB05212E396F3F2DC6A31B71C': 15,
 '2573A3CF633EBE6932A1E1010D5CD213': 16,
 'DA13A5C3763C212D9D68FC69102DE5E5': 17,
 '00304D7356D6C64481190D708D8F739C': 18,
 '7D11A7AA105DAB4D6799AF863369DB9C': 19,
 '23686A079CA538645BF6118A1EF51C8B': 20,
 'A5CFB818D79497B482B7225887DBD3AD': 21,
 '838A92D9F7EB57FB4A8B0C953A80C7EB': 22,
 '99CA116BF6AA65D70F3C78BEBADC51F0': 23,
 'D922D8FEA3EFAD3200455120B75BCEB8': 24,
 '159541FA269CA8A9CDB93658CAEC4CA2': 25,
 'E84BE2C963852FB065EE827F41A0A304': 26,
 '6B90065EA806B8523C0A6E56D7A961B2': 27,
 '4B55C45CD308068E4D0913DEF1043AD6': 28,
 'BAC6A3C2E18C26A77C99B41ECE1C738D': 29,
 '4CA37504EF8BA4352B03DCBA50E98A45': 30,
 '3228B1FB4BC92E81EF2FE35BDA86C540': 31,
 'D7C16BC3C9A5A633D6A3043A567C95A6': 32,
 '477ED2ED930405BF1DBF13F9BF973434': 33,
 '41776FB50B812A6775C2F8DEC92A9779': 34,
 'C1E99BF67DDA2227007DE8038FE32470': 35,
 'F70598172AC4514B1E6818EA361AD580': 36,
 '6744F8519308FD72D8C47BD45186303C': 37,
 '10C6C994C2AD434F9D49D4BE9CFBC613': 38,
 '89CE0912454AFE0A1B959569C37A5B8F': 39,
 '105008E45831ADE8AF1DB888319F422A': 40,
 'DE8A3755FCEDC549A408D7B1EB1A2C9F': 41,
 'BF04E736C599E9DE22F39F1DC157E1F1': 42,
 'CF304ED3CFC1ADD26720B97B39900FFD': 43,
 '59BE899EB83AAA19878738040F6828F0': 44,
 '3DF931B225B690508A63FD24133FA0E2': 45,
 '3AB05D6A4045A6C37D3E4566CFDFFE26': 46,
 '678E280656F6A0C0C23D5DFD46B85C14': 47,
 '440116720BC3A7957E216A77EE5C18CF': 48,
 'A3E4360031A7E05E9279F4D504EE18DD': 49,
 'C41F6D723AB5D14716D856DF9C000DED': 50,
 '7E18F69967284BB0601E88A114B8F7A9': 51,
 'F9D8F1DB5A398E1225A2C42E34A51DF6': 52,
 '914074E75CB398B5A2D81E1A51818CAA': 53,
 '5B210378BE9FFA3C90818C43B29B466B': 54,
 'F33767F7D7080003F403FDAB34FEB755': 55,
 'DC5C9FB3F0B3B740BAEE4F6049C2C7F1': 56,
 '3EA57373381A56822CBBC736169D0145': 57,
 '37342508F52BF4B62CCE3BA25460F9EB': 58,
 '7168CE9B777B76E4069A538DC5F28B6F': 59,
 '0BB2C843174730BA7D958C98B763A797': 60,
 'CDE47D81F953D800F760F1DE8AA754BA': 61,
 '9D831A0F3603A54732CCBDBF291D17B7': 62,
 '5F152815982885A996841493F2757D91': 63,
 '82C9890E4A7FC1F8730A3443C761143E': 64,
 '8C64085F46CD49FA5C80E72A35845185': 65}
MAP_LANG = cudf.DataFrame( cudf.Series(MAP_LANG) ).reset_index()
MAP_LANG.columns = ['language2','language']
MAP_LANG['language'] = MAP_LANG['language'].astype('int8')


def hashit_GPU(series):
    return series.hash_values()

def extract_hash_GPU(text, split_text='@', no=0):
    # NEED TO UPDATE TO FIND NAMES CONNECTED WITH UNDERSCORE
    text = text.str.lower()
    text_split = text.str.partition(split_text)
    for k in range(no):
        text_split = text_split[2].str.partition(split_text)
    word = text_split[2].str.partition(' ')[0].fillna('')
    word = clean_text_GPU( word )
    return hashit_GPU( word )

def clean_text_GPU(text):
    text = text+' '
    for s in ['!', '\?', ':', ';', '.', ',','']:
        text = text.str.replace(f'{s} ','')
    return(text)

def check_last_char_quest_GPU(tmp):
    # NEED TO UPDATE TO REMOVE HASH TAGS NAMES AND LINKS BEFORE SEARCHING
    tmp = tmp.copy().str.rstrip()
    end1 = tmp.str.get(-2) 
    end0 = tmp.str.get(-1)
    x = ((end1!='!')&(end0=='?')).astype('int8') * 1
    x += ((end1=='!')&(end0=='?')).astype('int8') * 2
    x += ((end1!='?')&(end0=='!')).astype('int8') * 3
    x += ((end1=='?')&(end0=='!')).astype('int8') * 4
    return x

def extract_feature(fn, tokenizer=None, test_tokens_dir='./test_tokens', test_proc3_dir='./test_proc3', 
                    submission=True, skip=0, chunk=3_000_000, part_num=0):
        
    # READ CHUNK OF TSV FROM DISK
    df = cudf.read_csv(fn, sep='\x01', header=None, skiprows=skip, nrows=chunk)
    if len(df)==0: return -1
    
    # ADD DUMMY TARGET COLUMNS
    print(f"fn = {fn}_{part_num}, df.shape = {df.shape}, skip = {skip}")
    if submission:
        df.columns = features_submission
        df['like'] = 0
        df['reply'] = 0
        df['retweet'] = 0
        df['retweet_comment'] = 0
    else:
        df.columns = features 
        
    # SAVE ORIGINAL TWEET_ID AND B_USER_ID
    df['tweet_id_org'] = df['tweet_id'].copy()
    df['b_user_id_org'] = df['b_user_id'].copy()  
    
    # CONVERT TARGETS TO INT8
    df['reply'] = df['reply'].notna().astype('int8')
    df['retweet'] = df['retweet'].notna().astype('int8')
    df['retweet_comment'] = df['retweet_comment'].notna().astype('int8')
    df['like'] = df['like'].notna().astype('int8')    
      
    #######################
    # TWEET TOKEN PROCESSING
    #######################
    
    df['tw_len_token'] = df['text_tokens'].str.count('\t').astype('int16')
    
    # MAKE TOKENIZER ON GPU
    if not tokenizer is None:
        tokenizer = cudf.DataFrame( list(tokenizer.get_vocab().keys()) )
        tokenizer = tokenizer.reset_index()
        tokenizer['index'] = tokenizer['index'].astype('int32')
        tokenizer.columns = ['token_id','token']
    else:
        tokenizer = cudf.read_parquet('bert-base-multilingual-cased.parquet')

    # PROCESS TRAIN TOKEN IDS
    tmp = df['text_tokens'].str.split('\t') 
    tmp = cudf.DataFrame( tmp.explode().astype('int32') ) 
    tmp['idx'] = cp.arange(len(tmp))
    tmp = tmp.reset_index()
    tmp.columns = ['row','token_id','idx']

    # GPU DECODE TOKEN IDS
    tmp = tmp.merge(tokenizer,how='left',on='token_id')
    tmp = tmp.sort_values('idx')
    tmp = tmp[['row','token']].groupby('row').collect()
    df['text'] = tmp.token.str.join(' ').str.replace(' ##','')
    del tmp
    
    # SAVE TOKEN IDS
    max_len = 48   
    tmp = df['text_tokens'] + "\t0"*max_len
    tmp = tmp.str.split('\t')
    tokens = cp.zeros((df.shape[0],max_len),dtype='int32')
    for k in range(max_len):
        tokens[:,k] = tmp.list.get(k).astype('int32').values        
    os.makedirs(test_tokens_dir, exist_ok=True)
    cp.save(os.path.join(test_tokens_dir , fn.split('/')[-1]+'_%i'%part_num ),tokens)
    del tmp, df['text_tokens']
    
    #######################
    # ENGINEER FEATURES
    ##########################################################################################   
    df['media'] = df['media'].fillna('')
    df['tw_len_media'] = df['media'].str.count('\t').astype('int8')
    df['tw_len_photo'] = df['media'].str.count('Photo').astype('int8')
    df['tw_len_video'] = df['media'].str.count('Video').astype('int8')
    df['tw_len_gif'] = df['media'].str.count('GIF').astype('int8')
    df['tw_len_quest'] = df['text'].str.count('\?').astype('int8')
    df['tw_count_capital_words'] = df['text'].str.count('[A-Z]{2,}').astype('int16') 
    df['tw_count_excl_quest_marks'] = df['text'].str.count('!|\?').astype('int16')
    df['tw_count_special1'] = df['text'].str.count('¶').astype('int16')
    df['tw_count_hash'] = df['text'].str.count('#').astype('int16')
    tmp = df['text'].str.replace('[sep]', '').str.partition('http')[0]
    df['tw_last_quest'] = check_last_char_quest_GPU(tmp) 
    
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].str.replace('http : / / t. co / ', 'http') 
    df['text'] = df['text'].str.replace('https : / / t. co / ', 'http') 
    df['text'] = df['text'].str[0:-5]
    df['text'] = df['text'].str.replace(' _ ', '_') 
    df['text'] = df['text'].str.replace('@ ', '@')
    df['text'] = df['text'].str.replace('# ', '#')
    
    df['tw_len_retweet'] = df['text'].str.count('retweet').astype('int8')    
    df['tw_isrt'] = (df['tweet_type']=='Retweet').astype('int8')
    df['text'] = df['text'].str.replace('\[cls\] rt @', '@')
    df['text'] = df['text'].str.replace('\[cls\] ', ' ')
    df['tw_len_rt'] = tmp.str.rstrip().str.count(' rt ').astype('int8')
    del tmp
    
    # SPLIT RETWEET TEXT AND ORIGINAL TEXT
    df['rt_text'] = ''
    tmp = df.loc[df.tw_isrt!=0,'text'].str.partition(':')
    df.loc[df.tw_isrt!=0,'rt_text'] = tmp[0]
    df.loc[df.tw_isrt!=0,'text'] = tmp[2]
    
    df['tw_count_at'] = df['text'].str.count('@').astype('int16')
    df['text'] = df['text'].str.replace('¶ ', ' ')
    
    df['rt_text'] = df['rt_text'].str.replace('¶ ', ' ')
    df['text'] = df['text'].str.strip()
    
    df['rt_text'] = df['rt_text'].str.strip()
    df['text'] = df['text'].str.split().str.join(' ')
    
    df['rt_text'] = df['rt_text'].str.split().str.join(' ')
    
    df['tw_count_words'] = df['text'].str.count(' ').fillna(0).astype('int16')
    df['tw_count_char']  = df['text'].str.len().fillna(0).astype('int16')
    df['tw_rt_count_words'] = df['rt_text'].str.count(' ').fillna(0).astype('int16')
    df['tw_rt_count_char']  = df['rt_text'].str.len().fillna(0).astype('int16')
    df['tw_original_user0'] = extract_hash_GPU(df['text'], no=0)
    df['tw_original_user1'] = extract_hash_GPU(df['text'], no=1)
    df['tw_original_user2'] = extract_hash_GPU(df['text'], no=2)
    df['tw_rt_user0'] = extract_hash_GPU(df['rt_text'], no=0)
    
    tmp = df['text'].str.partition(' ') # USING GPU MURMUR3 HASH INSTEAD OF CPU MD5 HASH
    df['tw_word0'] = hashit_GPU(tmp[0].fillna('')) 
    df['tw_word1'] = hashit_GPU(tmp[2].str.partition(' ')[0].fillna(''))
    df['tw_tweet'] = hashit_GPU(df['text'])
    
    #######################
    # ENGINEER MORE FEATURES
    ##########################################################################################    
    df['group'] = 0
    df['group'] = df['group'] + 1*(df['a_follower_count']>=240)
    df['group'] = df['group'] + 1*(df['a_follower_count']>=588)
    df['group'] = df['group'] + 1*(df['a_follower_count']>=1331)
    df['group'] = df['group'] + 1*(df['a_follower_count']>=3996)
    df['group'] = df['group'].astype('int8')
    
    df['date'] = cudf.to_datetime(df['timestamp'], unit='s')
    df['dt_day']  = df['date'].dt.day.astype('int8')
    df.loc[df.dt_day<4,'dt_day'] = 28 + df.loc[df.dt_day<4,'dt_day']
    df['dt_dow']  = df['date'].dt.weekday.astype('int8')
    df['dt_minute'] = df['date'].dt.hour.astype('int16') * 60 + df['date'].dt.minute.astype('int16')
    del df['date']
    
    df['hashtags'] = (df['hashtags']+'\t').fillna('')
    df['links'] = (df['links']+'\t').fillna('')
    df['domains'] = (df['domains']+'\t').fillna('')
    df['len_hashtags'] = df['hashtags'].str.count('\t').astype('int16')
    df['len_links'] = df['links'].str.count('\t').astype('int16')
    df['len_domains'] = df['domains'].str.count('\t').astype('int16')
    
    df['hashtags'] = df['hashtags'].str.split('\t').list.get(0).str.hex_to_int()
    df['hashtags'] = (df['hashtags']%2**32).astype('int32')
    df['links'] = df['links'].str.split('\t').list.get(0).str.hex_to_int()
    df['links'] = (df['links']%2**32).astype('int32')
    df['domains'] = df['domains'].str.split('\t').list.get(0).str.hex_to_int()
    df['domains'] = (df['domains']%2**32).astype('int32')
            
    # LABEL ENCODE MEDIA, TWEET TYPE, LANGUAGE
    df = df.rename({'media':'media2'},axis=1)
    df['media2'] = df['media2'].str[:9]
    df = df.merge(MAP_MEDIA,on='media2',how='left')
    del df['media2']
    df['tweet_type'] = df['tweet_type'].fillna('')
    df = df.rename({'tweet_type':'tweet_type2'},axis=1)
    df = df.merge(MAP_TYPE,on='tweet_type2',how='left')
    del df['tweet_type2']
    df['language'] = df['language'].fillna('')
    df = df.rename({'language':'language2'},axis=1)  
    df = df.merge(MAP_LANG,on='language2',how='left')
    del df['language2']
        
    df['timestamp'] = df['timestamp'].astype('int64') # WAS uint32 but RAPIDS doesn't do uint
    
    df.loc[ df.a_account_creation<0 ,'a_account_creation'] = 1138308613
    df['a_account_creation'] = 240*(df['a_account_creation'] - 1138308613)/(1139000000 - 1138308613) - 127
    df['a_account_creation'] = df['a_account_creation'].astype('int8')
    
    df.loc[ df.b_account_creation<0 ,'b_account_creation'] = 1138308613
    df['b_account_creation'] = 240*(df['b_account_creation'] - 1138308613)/(1139000000 - 1138308613) - 127
    df['b_account_creation'] = df['b_account_creation'].astype('int8')

    df['a_follower_count'] = df['a_follower_count'].astype('int32')
    df['a_following_count'] = df['a_following_count'].astype('int32')
    df['b_follower_count'] = df['b_follower_count'].astype('int32')
    df['b_following_count'] = df['b_following_count'].astype('int32')

    df['a_is_verified'] = df['a_is_verified'].astype('int8')
    df['b_is_verified'] = df['b_is_verified'].astype('int8')
    df['b_follows_a'] = df['b_follows_a'].astype('int8')
    
    df['tweet_id'] = df['tweet_id'].str[-16:].str.hex_to_int().astype('int64')
    df['a_user_id32'] = df['a_user_id'].str[-8:].str.hex_to_int().astype('int32')
    df['b_user_id32'] = df['b_user_id'].str[-8:].str.hex_to_int().astype('int32')    

    df['a_user_id'] = df['a_user_id'].str[-16:].str.hex_to_int().astype('int64')
    df['b_user_id'] = df['b_user_id'].str[-16:].str.hex_to_int().astype('int64')

    df['decline'] = 0
    df.loc[(df.dt_day==31)&(df.dt_minute > 720), 'decline'] = df.loc[(df.dt_day==31)&(df.dt_minute > 720), 'dt_minute'] - 720        
    df.loc[(df.dt_day==24)&(df.dt_minute > 720), 'decline'] = df.loc[(df.dt_day==24)&(df.dt_minute > 720), 'dt_minute'] - 720        
    df['decline'] = df['decline'].astype('int16')   
    
    del df['rt_text']; del df['tw_isrt']
    
    #######################
    # WRITE PROCESSED DATAFRAME TO PARQUET
    #######################   
    os.makedirs(test_proc3_dir, exist_ok=True)
    df.to_parquet(os.path.join(test_proc3_dir, fn.split('/')[-1] +'_%i.parquet'%part_num))
    lines_read = len(df)
    del df; gc.collect()
    
    return lines_read


############################################################################
# PROCESS TSV DATA, ENGINEER FEATURES, SAVE CONVERTED PARQUET
############################################################################
import time, glob
starttime = time.time()
targets = ['reply','retweet','retweet_comment','like']

tokenizer = None
#from transformers import BertTokenizer
#tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased') 

def process_test_set():
    print('Process_test_set')
    testfiles = glob.glob('./test/part*')
    print(testfiles)
    
    chunk = 3_000_000
    for fn in testfiles:
        i = 0
        skip = 0
        filesize = chunk
        while filesize==chunk:
            filesize = extract_feature(fn, tokenizer, skip=skip, chunk=chunk, part_num=i)
            print('-----------------------------------------------------------')
            skip += chunk 
            i += 1

if __name__ == "__main__":
    process_test_set() 
    print('elapsed:', time.time()-starttime) 
    print()
