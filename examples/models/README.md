
**MLP (aka 'NeuralSym' by Segler and Waller, 2017)**

We provide code and checkpoints for MLPs, which already yield decent performance.
We trained two models MLP and MLP++ (using a larger fingerprint) over three datasets USPTO-50k, USPTO-rt, and USPTO-rd, using both the training and validation set.
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
|MLP++|60.5|48.3|77.7|83.6|52.4|81.8|86.8|

**rt-1k**

|Model| mrr |top-1|top-5|top-10|maxfrag-1|maxfrag-5|maxfrag-10|mss|
| --- |---| --- | --- | --- | --- | --- | --- | --- |
|MLP| 41.1 |31.4|55.0|60.3|35.5|61.7|67.3 |43.0|
|MLP++|42.4|32.5|56.2|61.5|36.9|63.0|68.3| 44.2|
**rd-1k**

|Model| mrr|top-1|top-5|top-10|maxfrag-1|maxfrag-5|maxfrag-10|mss|
| --- |---| --- | --- | --- | --- | --- | --- |---|
|MLP| 37.5|28.8|49.6|54.6|33.4|56.6|61.7|29.7|
|MLP++|38.7|30.0|51.0|55.9|34.4|57.8|62.8|31.3|
**Trained Over USPTO-rt**

**rt-1k**

|Model| mrr | top-1 |top-5|top-10| maxfrag-1     | maxfrag-5 | maxfrag-10 |mss|
| --- |---|-------| --- | --- |---------------|----------|---------| ---|
|MLP| 48.9 | 37.4  |65.2|72.6|41.6|70.5|77.6|55.6|
|MLP++|49.3| 37.9  |65.6|72.1 |42.2|70.8|77.1 |54.8|
**rt**

|Model| mrr |top-1|top-5|top-10|maxfrag-1|maxfrag-5|maxfrag-10|mss|
| --- |---| --- | --- | --- | --- | --- | --- | ---|
|MLP| 48.8 |37.1|65.1|72.5|41.2|70.5|77.5|55.1|
|MLP++|49.8|38.2|65.7|72.7|42.4|71.2|77.9|55.6|

**Trained Over USPTO-rd**

**rd-1k**

|Model| mrr  |top-1|top-5|top-10|maxfrag-1|maxfrag-5|maxfrag-10| mss |
| --- |------| --- | --- | --- | --- | --- | --- |-----| 
|MLP| 51.5 |40.3|66.8|73.5|45.0|71.5|78.2|48.7|
|MLP++|52.3|41.4|67.5|73.5|45.9|72.9|78.7|47.8|
**rd**

|Model| mrr  |top-1|top-5|top-10|maxfrag-1|maxfrag-5|maxfrag-10|mss|
| --- |------| --- | --- | --- | --- | --- | --- | --- | 
|MLP| 52.0 |40.8|67.5|74.2|45.2|72.7|79.1|49.0|
|MLP++|52.7|41.4|68.2|74.4|45.9|73.7|79.5|49.1|
