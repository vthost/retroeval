
**MLP (aka 'NeuralSym' by Segler and Waller, 2017)**

We provide code and checkpoints for MLPs, which already yield decent performance.
We trained two models MLP and MLP++ (using a larger fingerprint, will be provided soon) over three datasets USPTO-50k, USPTO-rt, and USPTO-rd, using both the training and validation set.
The exact training configurations are given in [examples/configs](../../examples/configs).

Please note, the results vary slightly from the ones in our paper since:
- We updated the configurations.
- We used the cleaned USPTO-50k data from [rxn-ebm](https://github.com/coleygroup/rxn-ebm). 
- We used all templates also for the experiments over USPTO-rt and rd, for simplicity. In the paper, we dropped the ones occuring only once with MLP. 

The performance for these models is as follows.


**Trained Over USPTO-50k**

**USPTO-50k**

|Model| mrr  |top-1|top-5|top-10|maxfrag-1|maxfrag-5|maxfrag-10|
| --- |---| --- | --- | --- | --- | --- | --- | 
|MLP| 59.0 |46.4|76.3|82.9|50.0|80.3|86.3|


**rt-1k**

|Model| mrr |top-1|top-5|top-10|maxfrag-1|maxfrag-5|maxfrag-10|mss|
| --- |---| --- | --- | --- | --- | --- | --- | --- |
|MLP| 41.1 |31.4|55.0|60.3|35.5|61.7|67.3 |43.0|

**rd-1k**

|Model| mrr|top-1|top-5|top-10|maxfrag-1|maxfrag-5|maxfrag-10|mss|
| --- |---| --- | --- | --- | --- | --- | --- |---|
|MLP| 37.5|28.8|49.6|54.6|33.4|56.6|61.7|29.7|

**Trained Over USPTO-rt**

**rt-1k**

|Model| mrr |top-1|top-5|top-10| maxfrag-1     | maxfrag-5 | maxfrag-10 |mss|
| --- |---| --- | --- | --- |---------------|----------|------------| ---|
|MLP| 48.9 |37.4|65.2|72.6|41.6|70.5|77.6|55.6|

**rt**

|Model| mrr |top-1|top-5|top-10|maxfrag-1|maxfrag-5|maxfrag-10|mss|
| --- |---| --- | --- | --- | --- | --- | --- | ---|
|MLP| 48.8 |37.1|65.1|72.5|41.2|70.5|77.5|55.1|


**Trained Over USPTO-rd**

**rd-1k**

|Model| mrr  |top-1|top-5|top-10|maxfrag-1|maxfrag-5|maxfrag-10| mss |
| --- |------| --- | --- | --- | --- | --- | --- |-----| 
|MLP| 51.5 |40.3|66.8|73.5|45.0|71.5|78.2|48.7|

**rd**

|Model| mrr  |top-1|top-5|top-10|maxfrag-1|maxfrag-5|maxfrag-10|mss|
| --- |------| --- | --- | --- | --- | --- | --- | --- | 
|MLP| 52.0 |40.8|67.5|74.2|45.2|72.7|79.1|49.0|

