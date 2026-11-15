질문 500개, greedy. 값은 %이고 대괄호는 질문을 다시 뽑는 bootstrap으로 구한 95% 구간이다.

| 조건 | 유효한 JSON | key 이름이 answer | 정답 문자열이 들어 있음 | JSON 뒤에 다른 글이 없음 | 바뀐 가중치 수 |
| --- | --- | --- | --- | --- | --- |
| no edit | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 77.4 [73.8, 81.2] | 0.0 [0.0, 0.0] | 0 |
| instruction in the prompt | 99.6 [99.0, 100.0] | 99.6 [99.0, 100.0] | 75.4 [71.6, 79.0] | 99.6 [99.0, 100.0] | 0 |
| difference added at every position of layer 20, scale 1.0 (hook) | 31.8 [27.6, 36.2] | 4.6 [2.8, 6.4] | 48.8 [44.6, 53.4] | 29.4 [25.4, 33.6] | 0 |
| difference added at the answer start only, layer 22, scale 4.0 (hook) | 90.2 [87.4, 92.6] | 17.2 [14.0, 20.6] | 67.4 [63.2, 71.4] | 89.6 [86.8, 92.2] | 0 |
| differences added at the answer start only, all layers (hook) | 95.4 [93.6, 97.0] | 12.4 [9.6, 15.4] | 77.2 [73.4, 80.8] | 95.4 [93.6, 97.0] | 0 |
| differences added at the answer start only, every 4th layer (hook) | 86.8 [83.8, 89.6] | 7.4 [5.2, 9.6] | 78.0 [74.4, 81.6] | 86.6 [83.6, 89.4] | 0 |
| differences added at the answer start only, later half (hook) | 94.8 [92.8, 96.8] | 20.2 [16.8, 23.6] | 79.6 [76.0, 83.0] | 94.6 [92.4, 96.6] | 0 |
| differences added at the answer start only, earlier half (hook) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 78.6 [75.0, 82.2] | 0.0 [0.0, 0.0] | 0 |
| differences added at the answer start only, last quarter (hook) | 71.4 [67.4, 75.2] | 5.4 [3.4, 7.4] | 77.8 [74.2, 81.4] | 71.2 [67.2, 75.0] | 0 |
| differences added at the answer start only, last 2 layers (hook) | 81.2 [77.6, 84.2] | 10.4 [7.8, 13.0] | 63.4 [59.2, 67.6] | 81.0 [77.4, 84.2] | 0 |
| differences added at the answer start only, last layer (hook) | 74.0 [70.2, 77.8] | 10.4 [7.8, 13.0] | 76.8 [73.0, 80.6] | 73.8 [70.0, 77.6] | 0 |
| conditional edits, all layers (24 edits, weights only) | 95.8 [94.0, 97.4] | 14.6 [11.6, 17.8] | 76.6 [73.0, 80.2] | 95.8 [94.0, 97.4] | 73,505,712 |
| conditional edits, every 4th layer (6 edits, weights only) | 91.0 [88.4, 93.4] | 19.2 [15.8, 22.6] | 75.2 [71.4, 79.0] | 84.2 [81.0, 87.2] | 19,534,525 |
| conditional edits, later half (12 edits, weights only) | 92.8 [90.4, 95.0] | 19.2 [15.8, 22.6] | 75.4 [71.8, 79.0] | 92.6 [90.0, 94.8] | 40,565,810 |
| conditional edits, last quarter (6 edits, weights only) | 71.6 [67.6, 75.4] | 6.4 [4.4, 8.6] | 67.8 [63.6, 72.0] | 70.4 [66.4, 74.2] | 20,318,344 |
| conditional edits, every 8th layer (3 edits, weights only) | 54.0 [49.6, 58.2] | 9.6 [7.0, 12.0] | 58.6 [54.2, 62.8] | 54.0 [49.6, 58.2] | 10,029,741 |
| conditional edits, last 2 layers (2 edits, weights only) | 78.6 [75.2, 82.0] | 10.4 [7.8, 13.0] | 63.6 [59.4, 67.6] | 78.6 [75.2, 82.0] | 6,852,544 |
| conditional edits, last layer (1 edits, weights only) | 44.8 [40.4, 49.0] | 8.4 [6.0, 11.0] | 61.0 [56.8, 65.0] | 44.8 [40.4, 49.0] | 3,484,543 |
| control: every 4th layer, u replaced by random vectors of the same norm (weights only) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 71.6 [67.4, 75.4] | 0.0 [0.0, 0.0] | 19,479,261 |
| control: every 4th layer, entries of v shuffled (weights only) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 75.0 [71.2, 78.8] | 0.0 [0.0, 0.0] | 19,533,244 |
| control: every 4th layer, u taken from the lowercase instruction (weights only) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 69.0 [64.8, 73.2] | 0.0 [0.0, 0.0] | 19,235,814 |

유형별 유효한 JSON 비율(%)

| 조건 | fact | extract | arith | translate | sentiment |
| --- | --- | --- | --- | --- | --- |
| no edit | 0 | 0 | 0 | 0 | 0 |
| instruction in the prompt | 98 | 100 | 100 | 100 | 100 |
| difference added at every position of layer 20, scale 1.0 (hook) | 3 | 98 | 48 | 0 | 10 |
| difference added at the answer start only, layer 22, scale 4.0 (hook) | 99 | 100 | 54 | 98 | 100 |
| differences added at the answer start only, all layers (hook) | 95 | 100 | 99 | 84 | 99 |
| differences added at the answer start only, every 4th layer (hook) | 100 | 100 | 36 | 98 | 100 |
| differences added at the answer start only, later half (hook) | 84 | 100 | 97 | 93 | 100 |
| differences added at the answer start only, earlier half (hook) | 0 | 0 | 0 | 0 | 0 |
| differences added at the answer start only, last quarter (hook) | 62 | 100 | 0 | 95 | 100 |
| differences added at the answer start only, last 2 layers (hook) | 88 | 100 | 20 | 98 | 100 |
| differences added at the answer start only, last layer (hook) | 73 | 100 | 2 | 95 | 100 |
| conditional edits, all layers (24 edits, weights only) | 97 | 100 | 99 | 84 | 99 |
| conditional edits, every 4th layer (6 edits, weights only) | 95 | 100 | 85 | 95 | 80 |
| conditional edits, later half (12 edits, weights only) | 87 | 100 | 86 | 91 | 100 |
| conditional edits, last quarter (6 edits, weights only) | 65 | 100 | 5 | 95 | 93 |
| conditional edits, every 8th layer (3 edits, weights only) | 71 | 100 | 2 | 89 | 8 |
| conditional edits, last 2 layers (2 edits, weights only) | 88 | 100 | 20 | 85 | 100 |
| conditional edits, last layer (1 edits, weights only) | 58 | 100 | 0 | 64 | 2 |
| control: every 4th layer, u replaced by random vectors of the same norm (weights only) | 0 | 0 | 0 | 0 | 0 |
| control: every 4th layer, entries of v shuffled (weights only) | 0 | 0 | 0 | 0 | 0 |
| control: every 4th layer, u taken from the lowercase instruction (weights only) | 0 | 0 | 0 | 0 | 0 |
