### Motivation
TensorFlow evolves so fast that many code in this book can not run anymore because of deprecating of some api, so I fork this project to make sure that all code is runnable while my own learningship of TensorFlow
* `TensorFlow: 1.1.0`
* `Python: 3.5.3`

### API Update
```
tf.mul --> tf.multiply
tf.sub --> tf.subtract

tf.train.SummaryWriter --> tf.summary.FileWriter
tf.scalar_summary --> tf.summary.scalar
tf.merge_all_summaries --> tf.summary.merge_all

tf.initialize_all_variables --> tf.global_variables_initializer
tf.pack --> tf.stack
```

----------------------------------------------------------------------------------------

# `TensorFlow for Machine Intelligence`

![TensorFlow for Machine Intelligence book cover](img/book_cover.jpg)

Welcome to the official book repository for [_TensorFlow for Machine Intelligence_](https://bleedingedgepress.com/tensor-flow-for-machine-intelligence/)! Here, you'll find code from the book for easy testing on your own machine, as well as errata, and any additional content we can squeeze in down the line.

* **Code:** You'll find code for each chapter inside of the [chapters directory](https://github.com/backstopmedia/tensorflowbook/tree/master/chapters)
* **Errata:** Errata will be added to the [errata](https://github.com/backstopmedia/tensorflowbook/tree/master/errata) directory as they are discovered. Send in a pull request if you have errata to report!


