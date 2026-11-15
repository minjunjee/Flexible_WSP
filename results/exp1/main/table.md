질문 500개, greedy. 값은 %이고 대괄호는 질문을 다시 뽑는 bootstrap으로 구한 95% 구간이다.

| 조건 | 유효한 JSON | key 이름이 answer | 정답 문자열이 들어 있음 | JSON 뒤에 다른 글이 없음 | 바뀐 가중치 수 |
| --- | --- | --- | --- | --- | --- |
| no edit | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 77.6 [74.0, 81.4] | 0.0 [0.0, 0.0] | 0 |
| instruction in the prompt | 99.6 [99.0, 100.0] | 99.6 [99.0, 100.0] | 75.8 [72.0, 79.4] | 99.6 [99.0, 100.0] | 0 |
| difference added at every position of layer 20, scale 1.0 (hook) | 34.6 [30.4, 39.0] | 4.2 [2.6, 6.2] | 50.2 [45.8, 54.6] | 31.4 [27.2, 35.6] | 0 |
| difference added at the answer start only, layer 22, scale 1.0 (hook) | 90.0 [87.2, 92.6] | 24.4 [20.8, 28.2] | 77.8 [74.2, 81.4] | 89.8 [87.0, 92.4] | 0 |
| differences added at the answer start only, all layers (hook) | 96.2 [94.4, 97.6] | 12.8 [10.0, 15.8] | 76.6 [72.8, 80.2] | 96.2 [94.4, 97.6] | 0 |
| differences added at the answer start only, every 4th layer (hook) | 86.2 [83.2, 89.0] | 7.8 [5.6, 10.2] | 77.6 [74.0, 81.2] | 85.8 [82.6, 88.6] | 0 |
| differences added at the answer start only, later half (hook) | 94.6 [92.4, 96.4] | 20.2 [16.8, 23.8] | 78.8 [75.2, 82.4] | 94.4 [92.2, 96.2] | 0 |
| differences added at the answer start only, earlier half (hook) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 78.6 [75.0, 82.2] | 0.0 [0.0, 0.0] | 0 |
| differences added at the answer start only, last quarter (hook) | 70.6 [66.6, 74.4] | 4.6 [2.8, 6.4] | 77.0 [73.4, 80.8] | 70.6 [66.6, 74.4] | 0 |
| differences added at the answer start only, last 2 layers (hook) | 81.8 [78.2, 84.8] | 10.8 [8.2, 13.4] | 63.0 [58.8, 67.2] | 81.8 [78.2, 84.8] | 0 |
| differences added at the answer start only, last layer (hook) | 74.0 [70.0, 77.8] | 10.4 [7.8, 13.0] | 76.2 [72.4, 79.8] | 74.0 [70.0, 77.8] | 0 |
| conditional edits, all layers (24 edits, weights only) | 95.8 [94.0, 97.6] | 14.6 [11.6, 17.8] | 75.8 [72.0, 79.4] | 95.8 [94.0, 97.6] | 74,257,484 |
| conditional edits, every 4th layer (6 edits, weights only) | 81.2 [77.6, 84.4] | 22.4 [18.8, 26.0] | 75.8 [72.0, 79.4] | 71.6 [67.4, 75.4] | 19,230,771 |
| conditional edits, later half (12 edits, weights only) | 94.2 [92.0, 96.2] | 20.2 [16.8, 23.6] | 76.8 [73.2, 80.6] | 94.0 [91.8, 96.0] | 40,270,551 |
| conditional edits, last quarter (6 edits, weights only) | 76.2 [72.2, 79.8] | 12.0 [9.2, 14.8] | 69.4 [65.4, 73.4] | 69.8 [65.8, 73.6] | 20,284,999 |
| conditional edits, every 8th layer (3 edits, weights only) | 54.2 [49.8, 58.4] | 10.2 [7.4, 12.8] | 57.4 [53.2, 61.8] | 54.2 [49.8, 58.4] | 9,693,791 |
| conditional edits, last 2 layers (2 edits, weights only) | 80.8 [77.4, 84.0] | 10.8 [8.2, 13.4] | 64.0 [59.8, 68.0] | 80.8 [77.4, 84.0] | 7,015,113 |
| conditional edits, last layer (1 edits, weights only) | 51.6 [47.2, 55.8] | 9.8 [7.4, 12.4] | 59.4 [55.2, 63.6] | 51.6 [47.2, 55.8] | 3,601,541 |
| control: every 4th layer, u replaced by random vectors of the same norm (weights only) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 69.0 [65.0, 73.0] | 0.0 [0.0, 0.0] | 19,193,832 |
| control: every 4th layer, entries of v shuffled (weights only) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 77.4 [73.8, 81.2] | 0.0 [0.0, 0.0] | 19,227,623 |
| control: every 4th layer, u taken from the lowercase instruction (weights only) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 68.6 [64.4, 72.8] | 0.0 [0.0, 0.0] | 18,943,950 |

유형별 유효한 JSON 비율(%)

| 조건 | fact | extract | arith | translate | sentiment |
| --- | --- | --- | --- | --- | --- |
| no edit | 0 | 0 | 0 | 0 | 0 |
| instruction in the prompt | 98 | 100 | 100 | 100 | 100 |
| difference added at every position of layer 20, scale 1.0 (hook) | 1 | 98 | 57 | 0 | 17 |
| difference added at the answer start only, layer 22, scale 1.0 (hook) | 72 | 100 | 81 | 97 | 100 |
| differences added at the answer start only, all layers (hook) | 97 | 100 | 99 | 85 | 100 |
| differences added at the answer start only, every 4th layer (hook) | 100 | 100 | 33 | 98 | 100 |
| differences added at the answer start only, later half (hook) | 87 | 100 | 95 | 91 | 100 |
| differences added at the answer start only, earlier half (hook) | 0 | 0 | 0 | 0 | 0 |
| differences added at the answer start only, last quarter (hook) | 58 | 100 | 0 | 95 | 100 |
| differences added at the answer start only, last 2 layers (hook) | 88 | 100 | 23 | 98 | 100 |
| differences added at the answer start only, last layer (hook) | 74 | 100 | 1 | 95 | 100 |
| conditional edits, all layers (24 edits, weights only) | 98 | 100 | 100 | 82 | 99 |
| conditional edits, every 4th layer (6 edits, weights only) | 95 | 100 | 93 | 94 | 24 |
| conditional edits, later half (12 edits, weights only) | 87 | 100 | 93 | 91 | 100 |
| conditional edits, last quarter (6 edits, weights only) | 61 | 100 | 32 | 94 | 94 |
| conditional edits, every 8th layer (3 edits, weights only) | 75 | 100 | 0 | 90 | 6 |
| conditional edits, last 2 layers (2 edits, weights only) | 89 | 100 | 23 | 92 | 100 |
| conditional edits, last layer (1 edits, weights only) | 74 | 100 | 0 | 79 | 5 |
| control: every 4th layer, u replaced by random vectors of the same norm (weights only) | 0 | 0 | 0 | 0 | 0 |
| control: every 4th layer, entries of v shuffled (weights only) | 0 | 0 | 0 | 0 | 0 |
| control: every 4th layer, u taken from the lowercase instruction (weights only) | 0 | 0 | 0 | 0 | 0 |
