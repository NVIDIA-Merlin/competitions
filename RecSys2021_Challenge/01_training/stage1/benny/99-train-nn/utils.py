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

TWEET_CAT = [
    'media',
    'tweet_type',
    'language',
    'tw_len_media',
    'tw_len_photo',
    'tw_len_video',
    'tw_len_gif',
    'tw_last_quest',
    'len_hashtags',
    'len_links',
    'len_domains',
    'hashtags_',
    'domains_',
    'hour',
    'tw_rt_user0_',
    'tw_count_at',
    'tw_count_special1',
    'tw_len_quest',
    'tw_len_retweet',
    'tw_len_rt'
]
TWEET_NUM = [
    'tw_count_char',
    'tw_count_words',
    'tw_len_token'
]
A_USER_CAT = [
    'a_user_id_'
]
A_USER_NUM = [
    'a_follower_count',
    'a_following_count',
    'a_is_verified'
]
B_USER_CAT = [
    'b_user_id_'    
]
B_USER_NUM = [
    'b_follower_count',
    'b_following_count',
    'b_is_verified'
]
OTHERS_CAT = [
    'dt_dow'
]
OTHERS_NUM = [
    'b_follows_a'
]
TARGETS = [
    'reply', 
    'retweet',
    'retweet_comment', 
    'like'
]

NUM_LOG_COLS = [
    'a_follower_count',
    'a_following_count',
    'b_follower_count',
    'b_following_count',
    'tw_count_char',
    'tw_count_words',
    'tw_len_token'
]

NUM_STATS = {
    'a_follower_count': [9.235696768523635, 3.111560771160793],
 'a_following_count': [6.2578431551824485, 1.8617217780232085],
 'b_follower_count': [5.266129734214283, 1.6605541058218978],
 'b_following_count': [5.834243150040552, 1.1939579003693612],
 'tw_count_char': [4.322766427792716, 0.8206800778231492],
 'tw_count_words': [2.535984523111658, 0.9972112373754571],
 'tw_len_token': [3.6257291983777318, 0.6890229560424759],
 'a_ff_rate': [0.6012308582841046, 1.3345731364618305],
 'b_ff_rate': [1.748865708799117, 142.32999985661687],
     'TE_a_user_id_like': [0.984, 0.0],
 'TE_a_user_id_retweet': [0.967, 0.0],
 'TE_a_user_id_retweet_comment': [0.579, 0.0],
 'TE_b_is_verified_tweet_type_reply': [0.049, 0.006],
 'TE_b_is_verified_tweet_type_like': [0.484, 0.145],
 'TE_b_is_verified_tweet_type_retweet': [0.113, 0.055],
 'TE_b_is_verified_tweet_type_retweet_comment': [0.021, 0.006],
 'TE_b_user_id_reply': [0.964, 0.0],
 'TE_b_user_id_like': [0.983, 0.001],
 'TE_b_user_id_retweet': [0.951, 0.0],
 'TE_b_user_id_retweet_comment': [0.55, 0.0],
 'TE_b_user_id_a_user_id_reply': [0.909, 0.0],
 'TE_b_user_id_a_user_id_like': [0.99, 0.003],
 'TE_b_user_id_a_user_id_retweet': [0.987, 0.001],
 'TE_b_user_id_a_user_id_retweet_comment': [0.631, 0.0],
 'TE_b_user_id_tweet_type_language_reply': [0.981, 0.0],
 'TE_b_user_id_tweet_type_language_like': [0.989, 0.002],
 'TE_b_user_id_tweet_type_language_retweet': [0.978, 0.0],
 'TE_b_user_id_tweet_type_language_retweet_comment': [0.583, 0.0],
    'TE_a_user_id_reply': [0.882, 0.0],
 'TE_domains_language_b_follows_a_tweet_type_media_a_is_verified_reply': [0.643,
  0.0],
 'TE_domains_language_b_follows_a_tweet_type_media_a_is_verified_like': [0.955,
  0.0],
 'TE_domains_language_b_follows_a_tweet_type_media_a_is_verified_retweet': [0.922,
  0.0],
 'TE_domains_language_b_follows_a_tweet_type_media_a_is_verified_retweet_comment': [0.404,
  0.0],
 'TE_media_tweet_type_language_a_is_verified_b_is_verified_b_follows_a_reply': [0.259,
  0.0],
 'TE_media_tweet_type_language_a_is_verified_b_is_verified_b_follows_a_like': [0.823,
  0.031],
 'TE_media_tweet_type_language_a_is_verified_b_is_verified_b_follows_a_retweet': [0.722,
  0.004],
 'TE_media_tweet_type_language_a_is_verified_b_is_verified_b_follows_a_retweet_comment': [0.106,
  0.0],
 'TE_tw_original_user0_tweet_type_language_reply': [0.858, 0.0],
 'TE_tw_original_user0_tweet_type_language_like': [0.991, 0.003],
 'TE_tw_original_user0_tweet_type_language_retweet': [0.929, 0.0],
 'TE_tw_original_user0_tweet_type_language_retweet_comment': [0.546, 0.0]
}

NUM_COLS = [
    'a_ff_rate',
    'b_ff_rate'
]

NUM_TE = [
    'TE_a_user_id_reply',
 'TE_a_user_id_like',
 'TE_a_user_id_retweet',
 'TE_a_user_id_retweet_comment',
 'TE_b_is_verified_tweet_type_reply',
 'TE_b_is_verified_tweet_type_like',
 'TE_b_is_verified_tweet_type_retweet',
 'TE_b_is_verified_tweet_type_retweet_comment',
 'TE_b_user_id_reply',
 'TE_b_user_id_like',
 'TE_b_user_id_retweet',
 'TE_b_user_id_retweet_comment',
#  'TE_b_user_id_a_user_id_reply',
#  'TE_b_user_id_a_user_id_like',
#  'TE_b_user_id_a_user_id_retweet',
#  'TE_b_user_id_a_user_id_retweet_comment',
 'TE_b_user_id_tweet_type_language_reply',
 'TE_b_user_id_tweet_type_language_like',
 'TE_b_user_id_tweet_type_language_retweet',
 'TE_b_user_id_tweet_type_language_retweet_comment',
 'TE_domains_language_b_follows_a_tweet_type_media_a_is_verified_reply',
 'TE_domains_language_b_follows_a_tweet_type_media_a_is_verified_like',
 'TE_domains_language_b_follows_a_tweet_type_media_a_is_verified_retweet',
 'TE_domains_language_b_follows_a_tweet_type_media_a_is_verified_retweet_comment',
 'TE_media_tweet_type_language_a_is_verified_b_is_verified_b_follows_a_reply',
 'TE_media_tweet_type_language_a_is_verified_b_is_verified_b_follows_a_like',
 'TE_media_tweet_type_language_a_is_verified_b_is_verified_b_follows_a_retweet',
 'TE_media_tweet_type_language_a_is_verified_b_is_verified_b_follows_a_retweet_comment',
 'TE_tw_original_user0_tweet_type_language_reply',
 'TE_tw_original_user0_tweet_type_language_like',
 'TE_tw_original_user0_tweet_type_language_retweet',
 'TE_tw_original_user0_tweet_type_language_retweet_comment'
]

emb_shape = {
    'media': [16, 4],
    'tweet_type': [4, 2],
    'language': [72, 16],
    'tw_len_media': [8, 2],
    'tw_len_photo': [8, 2],
    'tw_len_video': [8, 2],
    'tw_len_gif': [8, 2],
    'tw_last_quest': [8, 2],
    'len_hashtags': [64, 4],
    'len_links': [64, 4],
    'len_domains': [64, 4],
    'dt_dow': [16, 2],
    'b_user_id_': [13993879, 16],
    'a_user_id_': [13993879, 16],
    'm_user_id_': [0,16],
    'hour': [25, 8],
    'tw_last_quest': [6, 2],
    'tw_count_at': [7, 2],
    'tw_count_special1': [7, 2],
    'tw_len_quest': [7, 2],
    'tw_len_retweet': [2, 2],
    'tw_len_rt': [2, 2]
}