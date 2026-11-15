질문 500개, greedy. 값은 %이고 대괄호는 질문을 다시 뽑는 bootstrap으로 구한 95% 구간이다.

| 조건 | 유효한 JSON | key 이름이 answer | 정답 문자열이 들어 있음 | JSON 뒤에 다른 글이 없음 | 바뀐 가중치 수 |
| --- | --- | --- | --- | --- | --- |
| no edit | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 81.0 [77.6, 84.4] | 0.0 [0.0, 0.0] | 0 |
| instruction in the prompt | 99.6 [99.0, 100.0] | 99.6 [99.0, 100.0] | 75.8 [72.0, 79.4] | 99.6 [99.0, 100.0] | 0 |
| difference added at every position of layer 20, scale 1.0 (hook) | 35.4 [31.2, 39.8] | 4.6 [3.0, 6.6] | 50.2 [45.8, 54.6] | 32.2 [28.0, 36.4] | 0 |
| difference added at the answer start only, layer 22, scale 4.0 (hook) | 97.4 [96.0, 98.6] | 19.6 [16.2, 23.2] | 74.4 [70.6, 78.2] | 97.0 [95.4, 98.4] | 0 |
| differences added at the answer start only, all layers (hook) | 98.4 [97.2, 99.4] | 13.0 [10.0, 16.0] | 76.6 [72.8, 80.2] | 98.4 [97.2, 99.4] | 0 |
| differences added at the answer start only, every 4th layer (hook) | 86.6 [83.6, 89.4] | 7.8 [5.6, 10.2] | 77.8 [74.2, 81.4] | 86.2 [83.0, 89.0] | 0 |
| differences added at the answer start only, later half (hook) | 96.8 [95.2, 98.2] | 20.4 [17.0, 24.0] | 79.4 [75.8, 83.2] | 96.4 [94.6, 97.8] | 0 |
| differences added at the answer start only, earlier half (hook) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 80.4 [76.8, 83.8] | 0.0 [0.0, 0.0] | 0 |
| differences added at the answer start only, last quarter (hook) | 71.4 [67.4, 75.2] | 4.8 [3.0, 6.6] | 78.0 [74.4, 81.8] | 71.2 [67.2, 75.0] | 0 |
| differences added at the answer start only, last 2 layers (hook) | 83.0 [79.6, 86.0] | 10.8 [8.2, 13.4] | 77.2 [73.6, 81.0] | 83.0 [79.6, 86.0] | 0 |
| differences added at the answer start only, last layer (hook) | 74.2 [70.2, 78.0] | 10.4 [7.8, 13.0] | 77.4 [73.6, 81.2] | 74.0 [70.0, 77.8] | 0 |
| conditional edits, all layers (24 edits, weights only) | 98.8 [97.8, 99.6] | 13.8 [10.8, 17.0] | 77.0 [73.2, 80.4] | 98.6 [97.6, 99.6] | 75,004,378 |
| conditional edits, every 4th layer (6 edits, weights only) | 77.8 [74.0, 81.2] | 21.4 [17.8, 25.0] | 74.0 [70.2, 77.8] | 68.2 [63.8, 72.2] | 19,378,814 |
| conditional edits, later half (12 edits, weights only) | 96.0 [94.2, 97.6] | 20.2 [16.8, 23.6] | 76.6 [73.0, 80.4] | 95.4 [93.4, 97.2] | 40,318,931 |
| conditional edits, last quarter (6 edits, weights only) | 68.0 [64.0, 72.0] | 5.6 [3.6, 7.6] | 71.4 [67.4, 75.4] | 67.8 [63.8, 71.8] | 20,279,781 |
| conditional edits, every 8th layer (3 edits, weights only) | 54.8 [50.4, 59.0] | 10.4 [7.8, 13.0] | 58.4 [54.2, 62.6] | 54.8 [50.4, 59.0] | 9,871,503 |
| conditional edits, last 2 layers (2 edits, weights only) | 83.2 [80.0, 86.2] | 11.0 [8.4, 13.6] | 78.0 [74.4, 81.8] | 83.2 [80.0, 86.2] | 7,030,493 |
| conditional edits, last layer (1 edits, weights only) | 45.0 [40.8, 49.2] | 8.2 [5.8, 10.6] | 62.2 [58.0, 66.2] | 45.0 [40.8, 49.2] | 3,497,685 |
| control: every 4th layer, u replaced by random vectors of the same norm (weights only) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 78.8 [75.2, 82.4] | 0.0 [0.0, 0.0] | 19,342,007 |
| control: every 4th layer, entries of v shuffled (weights only) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 78.8 [75.2, 82.4] | 0.0 [0.0, 0.0] | 19,377,084 |
| control: every 4th layer, u taken from the lowercase instruction (weights only) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 68.2 [64.2, 72.4] | 0.0 [0.0, 0.0] | 19,093,601 |

유형별 유효한 JSON 비율(%)

| 조건 | fact | extract | arith | translate | sentiment |
| --- | --- | --- | --- | --- | --- |
| no edit | 0 | 0 | 0 | 0 | 0 |
| instruction in the prompt | 98 | 100 | 100 | 100 | 100 |
| difference added at every position of layer 20, scale 1.0 (hook) | 1 | 99 | 59 | 0 | 18 |
| difference added at the answer start only, layer 22, scale 4.0 (hook) | 99 | 100 | 88 | 100 | 100 |
| differences added at the answer start only, all layers (hook) | 99 | 100 | 99 | 94 | 100 |
| differences added at the answer start only, every 4th layer (hook) | 100 | 100 | 33 | 100 | 100 |
| differences added at the answer start only, later half (hook) | 88 | 100 | 97 | 99 | 100 |
| differences added at the answer start only, earlier half (hook) | 0 | 0 | 0 | 0 | 0 |
| differences added at the answer start only, last quarter (hook) | 59 | 100 | 0 | 98 | 100 |
| differences added at the answer start only, last 2 layers (hook) | 88 | 100 | 29 | 98 | 100 |
| differences added at the answer start only, last layer (hook) | 74 | 100 | 1 | 96 | 100 |
| conditional edits, all layers (24 edits, weights only) | 99 | 100 | 100 | 95 | 100 |
| conditional edits, every 4th layer (6 edits, weights only) | 96 | 100 | 91 | 98 | 4 |
| conditional edits, later half (12 edits, weights only) | 86 | 100 | 96 | 98 | 100 |
| conditional edits, last quarter (6 edits, weights only) | 57 | 100 | 3 | 95 | 85 |
| conditional edits, every 8th layer (3 edits, weights only) | 72 | 100 | 2 | 92 | 8 |
| conditional edits, last 2 layers (2 edits, weights only) | 89 | 100 | 36 | 91 | 100 |
| conditional edits, last layer (1 edits, weights only) | 58 | 100 | 0 | 64 | 3 |
| control: every 4th layer, u replaced by random vectors of the same norm (weights only) | 0 | 0 | 0 | 0 | 0 |
| control: every 4th layer, entries of v shuffled (weights only) | 0 | 0 | 0 | 0 | 0 |
| control: every 4th layer, u taken from the lowercase instruction (weights only) | 0 | 0 | 0 | 0 | 0 |
