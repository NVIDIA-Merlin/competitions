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

import pandas as pd, cudf
import numpy as np
import gc, time
import os
print('cudf version',cudf.__version__)

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
start = time.time()

SUBMISSION = os.path.exists('./test/')

#####dtype = {0: str, 1: str, 2: float}
#sub0 = cudf.read_csv('results_benny_stage2.csv', header=None, dtype=str ).sort_values(['0','1']).reset_index(drop=True)
sub0 = cudf.read_parquet('results_benny_stage2.parquet', header=None)\
           .sort_values(['tweet_id_org','b_user_id_org']).reset_index(drop=True)
sub1 = cudf.read_parquet('results-chris-stage2.pq', header=None)\
           .sort_values(['tweet_id_org','b_user_id_org']).reset_index(drop=True)
print(sub0.shape, sub1.shape)

sub0.columns = ['tweet_id','b_user','reply','retweet','quote','fav']
sub1.columns = ['tweet_id','b_user','reply','retweet','quote','fav']
gc.collect()

for tgt in ['reply','retweet','quote','fav']:
    sub0[tgt] = sub0[tgt].astype(np.float32)
    sub1[tgt] = sub1[tgt].astype(np.float32)

print('Means:')
print(sub0[['reply','retweet','quote','fav']].mean(0))
print(sub1[['reply','retweet','quote','fav']].mean(0))
print('---------------------')


### Correlations
#for tgt in ['reply','retweet','quote','fav']:
#    print(tgt,
#          np.corrcoef(sub0[tgt].values, sub1[tgt].values)[0][1],
#    )


W = [0.5, 0.5, 0.5, 0.5]
for i, tgt in enumerate(['reply','retweet','quote','fav']):
    sub0[tgt] = (sub0[tgt] + sub1[tgt])/2
    
del sub1; gc.collect()

sub0.to_csv('results.csv', header=None, index=False)

print('Blend:')
print(sub0[['reply','retweet','quote','fav']].mean(0))

print('Done!')
end = time.time()
print('Final Blend Total Script Elapsed',end-start,'seconds')