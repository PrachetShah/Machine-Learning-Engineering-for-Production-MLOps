# Ungraded Lab: Feature Engineering Pipeline


In this lab, you will continue exploring [Tensorflow Transform](https://www.tensorflow.org/tfx/transform/get_started). This time, it will be in the context of a machine learning (ML) pipeline. In production-grade projects, you want to streamline tasks so you can more easily improve your model or find issues that may arise. [Tensorflow Extended (TFX)](https://www.tensorflow.org/tfx) provides components that work together to execute the most common steps in a machine learning project. If you want to dig deeper into the motivations behind TFX and the need for machine learning pipelines, you can read about it in [this paper](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/b500d77bc4f518a1165c0ab43c8fac5d2948bc14.pdf) and in this [blog post](https://blog.tensorflow.org/2020/09/brief-history-of-tensorflow-extended-tfx.html).

You will build end-to-end pipelines in future courses but for this one, you will only build up to the feature engineering part. Specifically, you will:

* ingest data from a base directory with `ExampleGen`
* compute the statistics of the training data with `StatisticsGen`
* infer a schema with `SchemaGen`
* detect anomalies in the evaluation data with `ExampleValidator`
* preprocess the data into features suitable for model training with `Transform`

If several steps mentioned above sound familiar, it's because the TFX components that deal with data validation and analysis (i.e. `StatisticsGen`, `SchemaGen`, `ExampleValidator`) uses [Tensorflow Data Validation (TFDV)](https://www.tensorflow.org/tfx/data_validation/get_started) under the hood. You're already familiar with this library from the exercises in Week 1 and for this week, you'll see how it fits within an ML pipeline.

The components you will use are the orange boxes highlighted in the figure below:

<img src='img/feature_eng_pipeline.png'>



## Setup

### Import packages

Let's begin by importing the required packages and modules. In case you want to replicate this in your local workstation, we used *Tensorflow v2.6* and *TFX v1.3.0*.


```python
import tensorflow as tf

from tfx import v1 as tfx

from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from google.protobuf.json_format import MessageToDict

import os
import pprint
pp = pprint.PrettyPrinter()
```

### Define paths

You will define a few global variables to indicate paths in the local workspace.


```python
# location of the pipeline metadata store
_pipeline_root = './pipeline/'

# directory of the raw data files
_data_root = './data/census_data'

# path to the raw training data
_data_filepath = os.path.join(_data_root, 'adult.data')
```

### Preview the  dataset

You will again be using the [Census Income dataset](https://archive.ics.uci.edu/ml/datasets/Adult) from the Week 1 ungraded lab so you can compare outputs when just using stand-alone TFDV and when using it under TFX. Just to remind, the data can be used to predict if an individual earns more than or less than 50k US Dollars annually. Here is the description of the features again: 


* **age**: continuous.
* **workclass**: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
* **fnlwgt**: continuous.
* **education**: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
* **education-num**: continuous.
* **marital-status**: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
* **occupation**: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
* **relationship**: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
* **race**: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
* **sex**: Female, Male.
* **capital-gain**: continuous.
* **capital-loss**: continuous.
* **hours-per-week**: continuous.
* **native-country**: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.


```python
# preview the first few rows of the CSV file
!head {_data_filepath}
```

    age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,label
    39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K
    50, Self-emp-not-inc, 83311, Bachelors, 13, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 13, United-States, <=50K
    38, Private, 215646, HS-grad, 9, Divorced, Handlers-cleaners, Not-in-family, White, Male, 0, 0, 40, United-States, <=50K
    53, Private, 234721, 11th, 7, Married-civ-spouse, Handlers-cleaners, Husband, Black, Male, 0, 0, 40, United-States, <=50K
    28, Private, 338409, Bachelors, 13, Married-civ-spouse, Prof-specialty, Wife, Black, Female, 0, 0, 40, Cuba, <=50K
    37, Private, 284582, Masters, 14, Married-civ-spouse, Exec-managerial, Wife, White, Female, 0, 0, 40, United-States, <=50K
    49, Private, 160187, 9th, 5, Married-spouse-absent, Other-service, Not-in-family, Black, Female, 0, 0, 16, Jamaica, <=50K
    52, Self-emp-not-inc, 209642, HS-grad, 9, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 45, United-States, >50K
    31, Private, 45781, Masters, 14, Never-married, Prof-specialty, Not-in-family, White, Female, 14084, 0, 50, United-States, >50K


### Create the Interactive Context

When pushing to production, you want to automate the pipeline execution using orchestrators such as [Apache Beam](https://beam.apache.org/) and [Kubeflow](https://www.kubeflow.org/). You will not be doing that just yet and will instead execute the pipeline from this notebook. When experimenting in a notebook environment, you will be *manually* executing the pipeline components (i.e. you are the orchestrator). For that, TFX provides the [Interactive Context](https://github.com/tensorflow/tfx/blob/master/tfx/orchestration/experimental/interactive/interactive_context.py) so you can step through each component and inspect its outputs.

You will initialize the `InteractiveContext` below. This will create a database in the `_pipeline_root` directory which the different components will use to save or get the state of the component executions. You will learn more about this in Week 3 when we discuss ML Metadata. For now, you can think of it as the data store that makes it possible for the different pipeline components to work together. 

*Note: You can configure the database to connect to but for this exercise, we will just use the default which is a newly created local sqlite file.* ***You will see the warning after running the cell below and you can safely ignore it.***


```python
# Initialize the InteractiveContext with a local sqlite file.
# If you leave `_pipeline_root` blank, then the db will be created in a temporary directory.
# You can safely ignore the warning about the missing config file.
context = InteractiveContext(pipeline_root=_pipeline_root)
```

    WARNING:absl:InteractiveContext metadata_connection_config not provided: using SQLite ML Metadata database at ./pipeline/metadata.sqlite.


## Run TFX components interactively

With that, you can now run the pipeline interactively. You will see how to do that as you go through the different components below.

### ExampleGen

You will start the pipeline with the [ExampleGen](https://www.tensorflow.org/tfx/guide/examplegen) component. This  will:

*   split the data into training and evaluation sets (by default: 2/3 train, 1/3 eval).
*   convert each data row into `tf.train.Example` format. This [protocol buffer](https://developers.google.com/protocol-buffers) is designed for Tensorflow operations and is used by the TFX components.
*   compress and save the data collection under the `_pipeline_root` directory for other components to access. These examples are stored in `TFRecord` format. This optimizes read and write operations within Tensorflow especially if you have a large collection of data.

Its constructor takes the path to your data source/directory. In our case, this is the `_data_root` path. The component supports several data sources such as CSV, tf.Record, and BigQuery. Since our data is a CSV file, we will use [CsvExampleGen](https://www.tensorflow.org/tfx/api_docs/python/tfx/components/CsvExampleGen) to ingest the data.

Run the cell below to instantiate `CsvExampleGen`.


```python
# Instantiate ExampleGen with the input CSV dataset
example_gen = tfx.components.CsvExampleGen(input_base=_data_root)
```

You can execute the component by calling the `run()` method of the `InteractiveContext`.


```python
# Execute the component
context.run(example_gen)
```



    WARNING:root:Make sure that locally built Python SDK docker image has Python 3.8 interpreter.





<style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7fd394736130</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">1</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">CsvExampleGen</span><span class="deemphasize"> at 0x7fd394736220</span></div><table class="attr-table"><tr><td class="attr-name">.inputs</td><td class = "attrvalue">{}</td></tr><tr><td class="attr-name">.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7fd394736250</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/CsvExampleGen/examples/1)<span class="deemphasize"> at 0x7fd394736610</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/CsvExampleGen/examples/1</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.exec_properties</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['input_base']</td><td class = "attrvalue">./data/census_data</td></tr><tr><td class="attr-name">['input_config']</td><td class = "attrvalue">{
  &quot;splits&quot;: [
    {
      &quot;name&quot;: &quot;single_split&quot;,
      &quot;pattern&quot;: &quot;*&quot;
    }
  ]
}</td></tr><tr><td class="attr-name">['output_config']</td><td class = "attrvalue">{
  &quot;split_config&quot;: {
    &quot;splits&quot;: [
      {
        &quot;hash_buckets&quot;: 2,
        &quot;name&quot;: &quot;train&quot;
      },
      {
        &quot;hash_buckets&quot;: 1,
        &quot;name&quot;: &quot;eval&quot;
      }
    ]
  }
}</td></tr><tr><td class="attr-name">['output_data_format']</td><td class = "attrvalue">6</td></tr><tr><td class="attr-name">['output_file_format']</td><td class = "attrvalue">5</td></tr><tr><td class="attr-name">['custom_config']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['range_config']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['span']</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">['version']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['input_fingerprint']</td><td class = "attrvalue">split:single_split,num_files:1,total_bytes:3974460,xor_checksum:1635233824,sum_checksum:1635233824</td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue">{}</td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7fd394736250</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/CsvExampleGen/examples/1)<span class="deemphasize"> at 0x7fd394736610</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/CsvExampleGen/examples/1</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr></table></div>



You will notice that an output cell showing the execution results is automatically shown. This metadata is recorded into the database created earlier. This allows you to keep track of your project runs. For example, if you run it again, you will notice the `.execution_id` incrementing.

The output of the components are called *artifacts* and you can see an example by navigating through  `.component.outputs > ['examples'] > Channel > ._artifacts > [0]` above. It shows information such as where the converted data is stored (`.uri`) and the splits generated (`.split_names`).

You can also examine the output artifacts programmatically with the code below.


```python
# get the artifact object
artifact = example_gen.outputs['examples'].get()[0]

# print split names and uri
print(f'split names: {artifact.split_names}')
print(f'artifact uri: {artifact.uri}')
```

    split names: ["train", "eval"]
    artifact uri: ./pipeline/CsvExampleGen/examples/1


If you're wondering , the `number` in `./pipeline/CsvExampleGen/examples/{number}` is the execution id associated with that dataset. If you restart the kernel of this workspace and re-run up to this cell, you will notice a new folder with a different id name created. This shows that TFX is keeping versions of your data so you can roll back if you want to investigate a particular execution.

As mentioned, the ingested data is stored in the directory shown in the `uri` field. It is also compressed using `gzip` and you can verify by running the cell below.


```python
# Get the URI of the output artifact representing the training examples
train_uri = os.path.join(artifact.uri, 'Split-train')

# See the contents of the `train` folder
!ls {train_uri}
```

    data_tfrecord-00000-of-00001.gz


In a notebook environment, it may be useful to examine a few examples of the data especially if you're still experimenting. Since the data collection is saved in [TFRecord format](https://www.tensorflow.org/tutorials/load_data/tfrecord), you will need to use methods that work with that data type. You will need to unpack the individual examples from the `TFRecord` file and format it for printing. Let's do that in the following cells:


```python
# Get the list of files in this directory (all compressed TFRecord files)
tfrecord_filenames = [os.path.join(train_uri, name)
                      for name in os.listdir(train_uri)]

# Create a `TFRecordDataset` to read these files
dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type="GZIP")
```


```python
# Define a helper function to get individual examples
def get_records(dataset, num_records):
    '''Extracts records from the given dataset.
    Args:
        dataset (TFRecordDataset): dataset saved by ExampleGen
        num_records (int): number of records to preview
    '''
    
    # initialize an empty list
    records = []
    
    # Use the `take()` method to specify how many records to get
    for tfrecord in dataset.take(num_records):
        
        # Get the numpy property of the tensor
        serialized_example = tfrecord.numpy()
        
        # Initialize a `tf.train.Example()` to read the serialized data
        example = tf.train.Example()
        
        # Read the example data (output is a protocol buffer message)
        example.ParseFromString(serialized_example)
        
        # convert the protocol bufffer message to a Python dictionary
        example_dict = (MessageToDict(example))
        
        # append to the records list
        records.append(example_dict)
        
    return records
```


```python
# Get 3 records from the dataset
sample_records = get_records(dataset, 3)

# Print the output
pp.pprint(sample_records)
```

    [{'features': {'feature': {'age': {'int64List': {'value': ['39']}},
                               'capital-gain': {'int64List': {'value': ['2174']}},
                               'capital-loss': {'int64List': {'value': ['0']}},
                               'education': {'bytesList': {'value': ['IEJhY2hlbG9ycw==']}},
                               'education-num': {'int64List': {'value': ['13']}},
                               'fnlwgt': {'int64List': {'value': ['77516']}},
                               'hours-per-week': {'int64List': {'value': ['40']}},
                               'label': {'bytesList': {'value': ['IDw9NTBL']}},
                               'marital-status': {'bytesList': {'value': ['IE5ldmVyLW1hcnJpZWQ=']}},
                               'native-country': {'bytesList': {'value': ['IFVuaXRlZC1TdGF0ZXM=']}},
                               'occupation': {'bytesList': {'value': ['IEFkbS1jbGVyaWNhbA==']}},
                               'race': {'bytesList': {'value': ['IFdoaXRl']}},
                               'relationship': {'bytesList': {'value': ['IE5vdC1pbi1mYW1pbHk=']}},
                               'sex': {'bytesList': {'value': ['IE1hbGU=']}},
                               'workclass': {'bytesList': {'value': ['IFN0YXRlLWdvdg==']}}}}},
     {'features': {'feature': {'age': {'int64List': {'value': ['50']}},
                               'capital-gain': {'int64List': {'value': ['0']}},
                               'capital-loss': {'int64List': {'value': ['0']}},
                               'education': {'bytesList': {'value': ['IEJhY2hlbG9ycw==']}},
                               'education-num': {'int64List': {'value': ['13']}},
                               'fnlwgt': {'int64List': {'value': ['83311']}},
                               'hours-per-week': {'int64List': {'value': ['13']}},
                               'label': {'bytesList': {'value': ['IDw9NTBL']}},
                               'marital-status': {'bytesList': {'value': ['IE1hcnJpZWQtY2l2LXNwb3VzZQ==']}},
                               'native-country': {'bytesList': {'value': ['IFVuaXRlZC1TdGF0ZXM=']}},
                               'occupation': {'bytesList': {'value': ['IEV4ZWMtbWFuYWdlcmlhbA==']}},
                               'race': {'bytesList': {'value': ['IFdoaXRl']}},
                               'relationship': {'bytesList': {'value': ['IEh1c2JhbmQ=']}},
                               'sex': {'bytesList': {'value': ['IE1hbGU=']}},
                               'workclass': {'bytesList': {'value': ['IFNlbGYtZW1wLW5vdC1pbmM=']}}}}},
     {'features': {'feature': {'age': {'int64List': {'value': ['38']}},
                               'capital-gain': {'int64List': {'value': ['0']}},
                               'capital-loss': {'int64List': {'value': ['0']}},
                               'education': {'bytesList': {'value': ['IEhTLWdyYWQ=']}},
                               'education-num': {'int64List': {'value': ['9']}},
                               'fnlwgt': {'int64List': {'value': ['215646']}},
                               'hours-per-week': {'int64List': {'value': ['40']}},
                               'label': {'bytesList': {'value': ['IDw9NTBL']}},
                               'marital-status': {'bytesList': {'value': ['IERpdm9yY2Vk']}},
                               'native-country': {'bytesList': {'value': ['IFVuaXRlZC1TdGF0ZXM=']}},
                               'occupation': {'bytesList': {'value': ['IEhhbmRsZXJzLWNsZWFuZXJz']}},
                               'race': {'bytesList': {'value': ['IFdoaXRl']}},
                               'relationship': {'bytesList': {'value': ['IE5vdC1pbi1mYW1pbHk=']}},
                               'sex': {'bytesList': {'value': ['IE1hbGU=']}},
                               'workclass': {'bytesList': {'value': ['IFByaXZhdGU=']}}}}}]


Now that `ExampleGen` has finished ingesting the data, the next step is data analysis.

### StatisticsGen
The [StatisticsGen](https://www.tensorflow.org/tfx/guide/statsgen) component computes statistics over your dataset for data analysis, as well as for use in downstream components (i.e. next steps in the pipeline). As mentioned earlier, this component uses TFDV under the hood so its output will be familiar to you.

`StatisticsGen` takes as input the dataset we just ingested using `CsvExampleGen`.


```python
# Instantiate StatisticsGen with the ExampleGen ingested dataset
statistics_gen = tfx.components.StatisticsGen(
    examples=example_gen.outputs['examples'])

# Execute the component
context.run(statistics_gen)
```

    WARNING:root:Make sure that locally built Python SDK docker image has Python 3.8 interpreter.





<style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7fd4fcdc2640</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">2</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">StatisticsGen</span><span class="deemphasize"> at 0x7fd4fcdcbd00</span></div><table class="attr-table"><tr><td class="attr-name">.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7fd394736250</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/CsvExampleGen/examples/1)<span class="deemphasize"> at 0x7fd394736610</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/CsvExampleGen/examples/1</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['statistics']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7fd4fcdcba00</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/StatisticsGen/statistics/2)<span class="deemphasize"> at 0x7fd4fcdb4f40</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/StatisticsGen/statistics/2</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.exec_properties</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['stats_options_json']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['exclude_splits']</td><td class = "attrvalue">[]</td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7fd394736250</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/CsvExampleGen/examples/1)<span class="deemphasize"> at 0x7fd394736610</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/CsvExampleGen/examples/1</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['statistics']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7fd4fcdcba00</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/StatisticsGen/statistics/2)<span class="deemphasize"> at 0x7fd4fcdb4f40</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/StatisticsGen/statistics/2</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr></table></div>



You can display the statistics with the `show()` method.

*Note: You can safely ignore the warning shown when running the cell below.*


```python
# Show the output statistics
context.show(statistics_gen.outputs['statistics'])
```


<b>Artifact at ./pipeline/StatisticsGen/statistics/2</b><br/><br/>



<div><b>'train' split:</b></div><br/>



<iframe id='facets-iframe' width="100%" height="500px"></iframe>
        <script>
        facets_iframe = document.getElementById('facets-iframe');
        facets_html = '<script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"><\/script><link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/master/facets-dist/facets-jupyter.html"><facets-overview proto-input="Cp5nCg5saHNfc3RhdGlzdGljcxDhqQEawQgQAiKvCAq4AgjhqQEYASABLQAAgD8ypAIaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQCABQOGpARAQGhMSCCBIUy1ncmFkGQAAAAAAY7tAGhgSDSBTb21lLWNvbGxlZ2UZAAAAAAAOs0AaFRIKIEJhY2hlbG9ycxkAAAAAAMarQBoTEgggTWFzdGVycxkAAAAAAPCRQBoVEgogQXNzb2Mtdm9jGQAAAAAAgIxAGhASBSAxMXRoGQAAAAAA+IhAGhYSCyBBc3NvYy1hY2RtGQAAAAAA6IVAGhASBSAxMHRoGQAAAAAA8IJAGhMSCCA3dGgtOHRoGQAAAAAAEHxAGhcSDCBQcm9mLXNjaG9vbBkAAAAAANB3QBoPEgQgOXRoGQAAAAAAwHVAGhASBSAxMnRoGQAAAAAAcHJAGhUSCiBEb2N0b3JhdGUZAAAAAABAcUAaExIIIDV0aC02dGgZAAAAAAAAa0AaExIIIDFzdC00dGgZAAAAAACAXUAaFRIKIFByZXNjaG9vbBkAAAAAAABCQCVE4xZBKpMDChMiCCBIUy1ncmFkKQAAAAAAY7tAChwIARABIg0gU29tZS1jb2xsZWdlKQAAAAAADrNAChkIAhACIgogQmFjaGVsb3JzKQAAAAAAxqtAChcIAxADIgggTWFzdGVycykAAAAAAPCRQAoZCAQQBCIKIEFzc29jLXZvYykAAAAAAICMQAoUCAUQBSIFIDExdGgpAAAAAAD4iEAKGggGEAYiCyBBc3NvYy1hY2RtKQAAAAAA6IVAChQIBxAHIgUgMTB0aCkAAAAAAPCCQAoXCAgQCCIIIDd0aC04dGgpAAAAAAAQfEAKGwgJEAkiDCBQcm9mLXNjaG9vbCkAAAAAANB3QAoTCAoQCiIEIDl0aCkAAAAAAMB1QAoUCAsQCyIFIDEydGgpAAAAAABwckAKGQgMEAwiCiBEb2N0b3JhdGUpAAAAAABAcUAKFwgNEA0iCCA1dGgtNnRoKQAAAAAAAGtAChcIDhAOIgggMXN0LTR0aCkAAAAAAIBdQAoZCA8QDyIKIFByZXNjaG9vbCkAAAAAAABCQEILCgllZHVjYXRpb24aoAMQAiKSAwq4AgjhqQEYASABLQAAgD8ypAIaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQCABQOGpARACGhESBiA8PTUwSxkAAAAAgCTQQBoQEgUgPjUwSxkAAAAAAE+0QCXwV7hAKikKESIGIDw9NTBLKQAAAACAJNBAChQIARABIgUgPjUwSykAAAAAAE+0QEIHCgVsYWJlbBryBRACItsFCrgCCOGpARgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAIAFA4akBEAcaHhITIE1hcnJpZWQtY2l2LXNwb3VzZRkAAAAAAJXDQBoZEg4gTmV2ZXItbWFycmllZBkAAAAAAOC7QBoUEgkgRGl2b3JjZWQZAAAAAAAap0AaFRIKIFNlcGFyYXRlZBkAAAAAALiEQBoTEgggV2lkb3dlZBkAAAAAAJiEQBohEhYgTWFycmllZC1zcG91c2UtYWJzZW50GQAAAAAAAHFAGh0SEiBNYXJyaWVkLUFGLXNwb3VzZRkAAAAAAAAwQCWpz3ZBKtcBCh4iEyBNYXJyaWVkLWNpdi1zcG91c2UpAAAAAACVw0AKHQgBEAEiDiBOZXZlci1tYXJyaWVkKQAAAAAA4LtAChgIAhACIgkgRGl2b3JjZWQpAAAAAAAap0AKGQgDEAMiCiBTZXBhcmF0ZWQpAAAAAAC4hEAKFwgEEAQiCCBXaWRvd2VkKQAAAAAAmIRACiUIBRAFIhYgTWFycmllZC1zcG91c2UtYWJzZW50KQAAAAAAAHFACiEIBhAGIhIgTWFycmllZC1BRi1zcG91c2UpAAAAAAAAMEBCEAoObWFyaXRhbC1zdGF0dXMaow4QAiKMDgq4AgjhqQEYASABLQAAgD8ypAIaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQCABQOGpARApGhkSDiBVbml0ZWQtU3RhdGVzGQAAAABA99JAGhISByBNZXhpY28ZAAAAAACQe0AaDRICID8ZAAAAAACAeUAaFxIMIFBoaWxpcHBpbmVzGQAAAAAAwGBAGhMSCCBHZXJtYW55GQAAAAAAQFdAGhISByBDYW5hZGEZAAAAAADAVUAaFxIMIFB1ZXJ0by1SaWNvGQAAAAAAAFJAGhESBiBJbmRpYRkAAAAAAEBQQBoTEgggRW5nbGFuZBkAAAAAAEBQQBoXEgwgRWwtU2FsdmFkb3IZAAAAAACAT0AaExIIIEphbWFpY2EZAAAAAACATUAaEBIFIEN1YmEZAAAAAAAATEAaERIGIENoaW5hGQAAAAAAAExAGhMSCCBWaWV0bmFtGQAAAAAAgEhAGhESBiBTb3V0aBkAAAAAAIBIQBoREgYgSXRhbHkZAAAAAACASEAaHhITIERvbWluaWNhbi1SZXB1YmxpYxkAAAAAAABHQBoREgYgSmFwYW4ZAAAAAACARUAaFBIJIENvbHVtYmlhGQAAAAAAgENAGhISByBUYWl3YW4ZAAAAAACAQkAlS2NUQSqbCAoZIg4gVW5pdGVkLVN0YXRlcykAAAAAQPfSQAoWCAEQASIHIE1leGljbykAAAAAAJB7QAoRCAIQAiICID8pAAAAAACAeUAKGwgDEAMiDCBQaGlsaXBwaW5lcykAAAAAAMBgQAoXCAQQBCIIIEdlcm1hbnkpAAAAAABAV0AKFggFEAUiByBDYW5hZGEpAAAAAADAVUAKGwgGEAYiDCBQdWVydG8tUmljbykAAAAAAABSQAoVCAcQByIGIEluZGlhKQAAAAAAQFBAChcICBAIIgggRW5nbGFuZCkAAAAAAEBQQAobCAkQCSIMIEVsLVNhbHZhZG9yKQAAAAAAgE9AChcIChAKIgggSmFtYWljYSkAAAAAAIBNQAoUCAsQCyIFIEN1YmEpAAAAAAAATEAKFQgMEAwiBiBDaGluYSkAAAAAAABMQAoXCA0QDSIIIFZpZXRuYW0pAAAAAACASEAKFQgOEA4iBiBTb3V0aCkAAAAAAIBIQAoVCA8QDyIGIEl0YWx5KQAAAAAAgEhACiIIEBAQIhMgRG9taW5pY2FuLVJlcHVibGljKQAAAAAAAEdAChUIERARIgYgSmFwYW4pAAAAAACARUAKGAgSEBIiCSBDb2x1bWJpYSkAAAAAAIBDQAoWCBMQEyIHIFRhaXdhbikAAAAAAIBCQAoZCBQQFCIKIEd1YXRlbWFsYSkAAAAAAIBCQAoWCBUQFSIHIFBvbGFuZCkAAAAAAABCQAoUCBYQFiIFIElyYW4pAAAAAAAAPkAKFQgXEBciBiBIYWl0aSkAAAAAAAA8QAoYCBgQGCIJIFBvcnR1Z2FsKQAAAAAAADtAChYIGRAZIgcgRnJhbmNlKQAAAAAAADhAChcIGhAaIgggRWN1YWRvcikAAAAAAAA2QAoWCBsQGyIHIEdyZWVjZSkAAAAAAAA1QAoZCBwQHCIKIE5pY2FyYWd1YSkAAAAAAAA0QAoUCB0QHSIFIFBlcnUpAAAAAAAAM0AKFwgeEB4iCCBJcmVsYW5kKQAAAAAAADJAChQIHxAfIgUgSG9uZykAAAAAAAAsQAoYCCAQICIJIENhbWJvZGlhKQAAAAAAACxAChoIIRAhIgsgWXVnb3NsYXZpYSkAAAAAAAAqQAofCCIQIiIQIFRyaW5hZGFkJlRvYmFnbykAAAAAAAAqQAoYCCMQIyIJIFRoYWlsYW5kKQAAAAAAACpAChcIJBAkIgggSHVuZ2FyeSkAAAAAAAAmQAoqCCUQJSIbIE91dGx5aW5nLVVTKEd1YW0tVVNWSS1ldGMpKQAAAAAAACRAChQIJhAmIgUgTGFvcykAAAAAAAAkQAoYCCcQJyIJIFNjb3RsYW5kKQAAAAAAACJAChgIKBAoIgkgSG9uZHVyYXMpAAAAAAAAIEBCEAoObmF0aXZlLWNvdW50cnkasgkQAiKfCQq4AgjhqQEYASABLQAAgD8ypAIaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQCABQOGpARAPGhsSECBFeGVjLW1hbmFnZXJpYWwZAAAAAABupUAaGhIPIFByb2Ytc3BlY2lhbHR5GQAAAAAAbKVAGhgSDSBDcmFmdC1yZXBhaXIZAAAAAABWpUAaGBINIEFkbS1jbGVyaWNhbBkAAAAAAMqjQBoREgYgU2FsZXMZAAAAAADeokAaGRIOIE90aGVyLXNlcnZpY2UZAAAAAAAcoUAaHRISIE1hY2hpbmUtb3AtaW5zcGN0GQAAAAAAXJVAGg0SAiA/GQAAAAAAPJNAGhwSESBUcmFuc3BvcnQtbW92aW5nGQAAAAAAZJBAGh0SEiBIYW5kbGVycy1jbGVhbmVycxkAAAAAADCMQBobEhAgRmFybWluZy1maXNoaW5nGQAAAAAACIVAGhgSDSBUZWNoLXN1cHBvcnQZAAAAAADogkAaGxIQIFByb3RlY3RpdmUtc2VydhkAAAAAAFB7QBobEhAgUHJpdi1ob3VzZS1zZXJ2GQAAAAAAQFpAGhgSDSBBcm1lZC1Gb3JjZXMZAAAAAAAAGEAlOWtTQSrJAwobIhAgRXhlYy1tYW5hZ2VyaWFsKQAAAAAAbqVACh4IARABIg8gUHJvZi1zcGVjaWFsdHkpAAAAAABspUAKHAgCEAIiDSBDcmFmdC1yZXBhaXIpAAAAAABWpUAKHAgDEAMiDSBBZG0tY2xlcmljYWwpAAAAAADKo0AKFQgEEAQiBiBTYWxlcykAAAAAAN6iQAodCAUQBSIOIE90aGVyLXNlcnZpY2UpAAAAAAAcoUAKIQgGEAYiEiBNYWNoaW5lLW9wLWluc3BjdCkAAAAAAFyVQAoRCAcQByICID8pAAAAAAA8k0AKIAgIEAgiESBUcmFuc3BvcnQtbW92aW5nKQAAAAAAZJBACiEICRAJIhIgSGFuZGxlcnMtY2xlYW5lcnMpAAAAAAAwjEAKHwgKEAoiECBGYXJtaW5nLWZpc2hpbmcpAAAAAAAIhUAKHAgLEAsiDSBUZWNoLXN1cHBvcnQpAAAAAADogkAKHwgMEAwiECBQcm90ZWN0aXZlLXNlcnYpAAAAAABQe0AKHwgNEA0iECBQcml2LWhvdXNlLXNlcnYpAAAAAABAWkAKHAgOEA4iDSBBcm1lZC1Gb3JjZXMpAAAAAAAAGEBCDAoKb2NjdXBhdGlvbhrUBBACIscECrgCCOGpARgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAIAFA4akBEAUaERIGIFdoaXRlGQAAAADAHNJAGhESBiBCbGFjaxkAAAAAAFigQBoeEhMgQXNpYW4tUGFjLUlzbGFuZGVyGQAAAAAAOIZAGh4SEyBBbWVyLUluZGlhbi1Fc2tpbW8ZAAAAAACgaUAaERIGIE90aGVyGQAAAAAAwGVAJWiJ0UAqiQEKESIGIFdoaXRlKQAAAADAHNJAChUIARABIgYgQmxhY2spAAAAAABYoEAKIggCEAIiEyBBc2lhbi1QYWMtSXNsYW5kZXIpAAAAAAA4hkAKIggDEAMiEyBBbWVyLUluZGlhbi1Fc2tpbW8pAAAAAACgaUAKFQgEEAQiBiBPdGhlcikAAAAAAMBlQEIGCgRyYWNlGoYFEAIi8QQKuAII4akBGAEgAS0AAIA/MqQCGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAgAUDhqQEQBhoTEgggSHVzYmFuZBkAAAAAAE7BQBoZEg4gTm90LWluLWZhbWlseRkAAAAAAI+1QBoVEgogT3duLWNoaWxkGQAAAAAAqqpAGhUSCiBVbm1hcnJpZWQZAAAAAAC2oUAaEBIFIFdpZmUZAAAAAADQj0AaGhIPIE90aGVyLXJlbGF0aXZlGQAAAAAAYIRAJSncIUEqoAEKEyIIIEh1c2JhbmQpAAAAAABOwUAKHQgBEAEiDiBOb3QtaW4tZmFtaWx5KQAAAAAAj7VAChkIAhACIgogT3duLWNoaWxkKQAAAAAAqqpAChkIAxADIgogVW5tYXJyaWVkKQAAAAAAtqFAChQIBBAEIgUgV2lmZSkAAAAAANCPQAoeCAUQBSIPIE90aGVyLXJlbGF0aXZlKQAAAAAAYIRAQg4KDHJlbGF0aW9uc2hpcBqgAxACIpQDCrgCCOGpARgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAIAFA4akBEAIaEBIFIE1hbGUZAAAAAABwzEAaEhIHIEZlbWFsZRkAAAAAAAG8QCWJHbVAKioKECIFIE1hbGUpAAAAAABwzEAKFggBEAEiByBGZW1hbGUpAAAAAAABvEBCBQoDc2V4GqMGEAIikQYKuAII4akBGAEgAS0AAIA/MqQCGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAgAUDhqQEQCRoTEgggUHJpdmF0ZRkAAAAAgLHNQBocEhEgU2VsZi1lbXAtbm90LWluYxkAAAAAAGyaQBoVEgogTG9jYWwtZ292GQAAAAAAbJVAGg0SAiA/GQAAAAAAMJNAGhUSCiBTdGF0ZS1nb3YZAAAAAAA4i0AaGBINIFNlbGYtZW1wLWluYxkAAAAAAICGQBoXEgwgRmVkZXJhbC1nb3YZAAAAAADIg0AaFxIMIFdpdGhvdXQtcGF5GQAAAAAAACJAGhgSDSBOZXZlci13b3JrZWQZAAAAAAAACEAlt6INQSr2AQoTIgggUHJpdmF0ZSkAAAAAgLHNQAogCAEQASIRIFNlbGYtZW1wLW5vdC1pbmMpAAAAAABsmkAKGQgCEAIiCiBMb2NhbC1nb3YpAAAAAABslUAKEQgDEAMiAiA/KQAAAAAAMJNAChkIBBAEIgogU3RhdGUtZ292KQAAAAAAOItAChwIBRAFIg0gU2VsZi1lbXAtaW5jKQAAAAAAgIZAChsIBhAGIgwgRmVkZXJhbC1nb3YpAAAAAADIg0AKGwgHEAciDCBXaXRob3V0LXBheSkAAAAAAAAiQAocCAgQCCINIE5ldmVyLXdvcmtlZCkAAAAAAAAIQEILCgl3b3JrY2xhc3MavgcatAcKuAII4akBGAEgAS0AAIA/MqQCGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAgAUDhqQERRe8La8JKQ0AZ2LxISFZhK0ApAAAAAAAAMUAxAAAAAACAQkA5AAAAAACAVkBCogIaGwkAAAAAAAAxQBHNzMzMzEw4QCG0WfW5upatQBobCc3MzMzMTDhAEZqZmZmZmT9AIR3r4jZ6m65AGhsJmpmZmZmZP0ARMzMzMzNzQ0Ah0LNZ9dkdr0AaGwkzMzMzM3NDQBGamZmZmRlHQCEWak3zrsuvQBobCZqZmZmZGUdAEQAAAAAAwEpAISsYldTpwqRAGhsJAAAAAADASkARZmZmZmZmTkAhnjws1FpDm0AaGwlmZmZmZmZOQBFmZmZmZgZRQCFns+pz9byQQBobCWZmZmZmBlFAEZqZmZmZ2VJAISRj7lpCx3RAGhsJmpmZmZnZUkARzczMzMysVEAhWOXQIlvlXkAaGwnNzMzMzKxUQBEAAAAAAIBWQCEeY+5aQsdEQEKkAhobCQAAAAAAADFAEQAAAAAAADZAIc3MzMzM+aBAGhsJAAAAAAAANkARAAAAAAAAOkAhzczMzMz5oEAaGwkAAAAAAAA6QBEAAAAAAAA9QCHNzMzMzPmgQBobCQAAAAAAAD1AEQAAAAAAgEBAIc3MzMzM+aBAGhsJAAAAAACAQEARAAAAAACAQkAhzczMzMz5oEAaGwkAAAAAAIBCQBEAAAAAAIBEQCHNzMzMzPmgQBobCQAAAAAAgERAEQAAAAAAgEZAIc3MzMzM+aBAGhsJAAAAAACARkARAAAAAAAASUAhzczMzMz5oEAaGwkAAAAAAABJQBEAAAAAAABNQCHNzMzMzPmgQBobCQAAAAAAAE1AEQAAAAAAgFZAIc3MzMzM+aBAIAFCBQoDYWdlGoQGGvEFCrgCCOGpARgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAIAFA4akBESoyHp3oiJBAGfnkVxj3RLxAIM2bATkAAAAA8Gn4QEKZAhoSETMzMzPzh8NAIWT8SRnRuNRAGhsJMzMzM/OHw0ARMzMzM/OH00Ahw4P+Tb1OdkAaGwkzMzMz84fTQBHMzMzM7EvdQCEa4xfrBGlGQBobCczMzMzsS91AETMzMzPzh+NAIUkyH8D8FQhAGhsJMzMzM/OH40ARAAAAAPBp6EAhSTIfwPwVCEAaGwkAAAAA8GnoQBHMzMzM7EvtQCFEMh/A/BUIQBobCczMzMzsS+1AEc3MzMz0FvFAIU0yH8D8FQhAGhsJzczMzPQW8UARMzMzM/OH80AhRDIfwPwVCEAaGwkzMzMz84fzQBGZmZmZ8fj1QCFEMh/A/BUIQBobCZmZmZnx+PVAEQAAAADwafhAIUbCd6RPe1ZAQnkaCSHNzMzMzPmgQBoJIc3MzMzM+aBAGgkhzczMzMz5oEAaCSHNzMzMzPmgQBoJIc3MzMzM+aBAGgkhzczMzMz5oEAaCSHNzMzMzPmgQBoJIc3MzMzM+aBAGgkhzczMzMz5oEAaEhEAAAAA8Gn4QCHNzMzMzPmgQCABQg4KDGNhcGl0YWwtZ2FpbhqEBhrxBQq4AgjhqQEYASABLQAAgD8ypAIaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQCABQOGpARGKR0gpUOxVQBn6pAnBbS95QCDioQE5AAAAAAAEsUBCmQIaEhGamZmZmTl7QCEG6qZGvzrUQBobCZqZmZmZOXtAEZqZmZmZOYtAIS8bi6+aYSxAGhsJmpmZmZk5i0ARNDMzMzNrlEAhkz7JJtjZMUAaGwk0MzMzM2uUQBGamZmZmTmbQCHm0CLbuUtzQBobCZqZmZmZOZtAEQAAAAAABKFAITtJyY0luYBAGhsJAAAAAAAEoUARNDMzMzNrpEAhcLGDHpVBXUAaGwk0MzMzM2ukQBFnZmZmZtKnQCGQIWQAnBEVQBobCWdmZmZm0qdAEZqZmZmZOatAIZAhZACcERVAGhsJmpmZmZk5q0ARzczMzMygrkAhkCFkAJwRFUAaGwnNzMzMzKCuQBEAAAAAAASxQCGQIWQAnBEVQEJ5GgkhzczMzMz5oEAaCSHNzMzMzPmgQBoJIc3MzMzM+aBAGgkhzczMzMz5oEAaCSHNzMzMzPmgQBoJIc3MzMzM+aBAGgkhzczMzMz5oEAaCSHNzMzMzPmgQBoJIc3MzMzM+aBAGhIRAAAAAAAEsUAhzczMzMz5oEAgAUIOCgxjYXBpdGFsLWxvc3MayAcatAcKuAII4akBGAEgAS0AAIA/MqQCGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAgAUDhqQER4IdTDkAjJEAZJed21hOgBEApAAAAAAAA8D8xAAAAAAAAJEA5AAAAAAAAMEBCogIaGwkAAAAAAADwPxEAAAAAAAAEQCEpXI/C9V5kQBobCQAAAAAAAARAEQAAAAAAABBAIWzn+6nxhGxAGhsJAAAAAAAAEEARAAAAAAAAFkAhqMZLNwkbiEAaGwkAAAAAAAAWQBEAAAAAAAAcQCHByqFFNlqDQBobCQAAAAAAABxAEQAAAAAAACFAITvfT41XzpBAGhsJAAAAAAAAIUARAAAAAAAAJEAhnxov3VR1u0AaGwkAAAAAAAAkQBEAAAAAAAAnQCHvp8ZLx562QBobCQAAAAAAACdAEQAAAAAAACpAIdejcD2KEYZAGhsJAAAAAAAAKkARAAAAAAAALUAh8/3UeJlKskAaGwkAAAAAAAAtQBEAAAAAAAAwQCFLN4lB4LWEQEKkAhobCQAAAAAAAPA/EQAAAAAAABxAIc3MzMzM+aBAGhsJAAAAAAAAHEARAAAAAAAAIkAhzczMzMz5oEAaGwkAAAAAAAAiQBEAAAAAAAAiQCHNzMzMzPmgQBobCQAAAAAAACJAEQAAAAAAACJAIc3MzMzM+aBAGhsJAAAAAAAAIkARAAAAAAAAJEAhzczMzMz5oEAaGwkAAAAAAAAkQBEAAAAAAAAkQCHNzMzMzPmgQBobCQAAAAAAACRAEQAAAAAAACZAIc3MzMzM+aBAGhsJAAAAAAAAJkARAAAAAAAAKkAhzczMzMz5oEAaGwkAAAAAAAAqQBEAAAAAAAAqQCHNzMzMzPmgQBobCQAAAAAAACpAEQAAAAAAADBAIc3MzMzM+aBAIAFCDwoNZWR1Y2F0aW9uLW51bRrBBxq0Bwq4AgjhqQEYASABLQAAgD8ypAIaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQCABQOGpARG0eWbG+C0HQRn/BaOBK+X5QCkAAAAAgP7HQDEAAAAAoMUFQTkAAAAAoac2QUKiAhobCQAAAACA/sdAEQAAAAA4eQNBIQtydoK0GcFAGhsJAAAAADh5A0ERAAAAAES5EkEhZFnkmn39w0AaGwkAAAAARLkSQREAAAAA7LUbQSEPRzJHfj2iQBobCQAAAADstRtBEQAAAABKWSJBIYPzD7p053JAGhsJAAAAAEpZIkERAAAAAJ7XJkEh2g1KS0BeUEAaGwkAAAAAntcmQREAAAAA8lUrQSFBlWKzjfcwQBobCQAAAADyVStBEQAAAABG1C9BIQSVuy/H9BJAGhsJAAAAAEbUL0ERAAAAAE0pMkEhBJW7L8f0EkAaGwkAAAAATSkyQREAAAAAd2g0QSEElbsvx/QSQBobCQAAAAB3aDRBEQAAAAChpzZBIQSVuy/H9BJAQqQCGhsJAAAAAID+x0ARAAAAAOBE8EAhzczMzMz5oEAaGwkAAAAA4ETwQBEAAAAAEBD6QCHNzMzMzPmgQBobCQAAAAAQEPpAEQAAAACg+/9AIc3MzMzM+aBAGhsJAAAAAKD7/0ARAAAAAMBfA0EhzczMzMz5oEAaGwkAAAAAwF8DQREAAAAAoMUFQSHNzMzMzPmgQBobCQAAAACgxQVBEQAAAABg9wdBIc3MzMzM+aBAGhsJAAAAAGD3B0ERAAAAAPjTCkEhzczMzMz5oEAaGwkAAAAA+NMKQREAAAAAyLgPQSHNzMzMzPmgQBobCQAAAADIuA9BEQAAAADUFxRBIc3MzMzM+aBAGhsJAAAAANQXFEERAAAAAKGnNkEhzczMzMz5oEAgAUIICgZmbmx3Z3QayQcatAcKuAII4akBGAEgAS0AAIA/MqQCGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzPmgQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzM+aBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMz5oEAgAUDhqQERWzQk34QyREAZpiLxnCejKEApAAAAAAAA8D8xAAAAAAAAREA5AAAAAADAWEBCogIaGwkAAAAAAADwPxGamZmZmZklQCGWsgxxrGt+QBobCZqZmZmZmSVAEZqZmZmZmTRAIRJhw9OrJ5dAGhsJmpmZmZmZNEARZ2ZmZmZmPkAhIkp7gy9PmEAaGwlnZmZmZmY+QBGamZmZmRlEQCGlcD0KJyHHQBobCZqZmZmZGURAEQAAAAAAAElAIZ2AJsJGQ6BAGhsJAAAAAAAASUARZ2ZmZmbmTUAhrthfds8qpEAaGwlnZmZmZuZNQBFnZmZmZmZRQCFBz2bVJ16SQBobCWdmZmZmZlFAEZqZmZmZ2VNAIbB3f7xX/HJAGhsJmpmZmZnZU0ARzczMzMxMVkAhnnb4a7JCYEAaGwnNzMzMzExWQBEAAAAAAMBYQCHbRgN4C3RWQEKkAhobCQAAAAAAAPA/EQAAAAAAADhAIc3MzMzM+aBAGhsJAAAAAAAAOEARAAAAAACAQUAhzczMzMz5oEAaGwkAAAAAAIBBQBEAAAAAAABEQCHNzMzMzPmgQBobCQAAAAAAAERAEQAAAAAAAERAIc3MzMzM+aBAGhsJAAAAAAAAREARAAAAAAAAREAhzczMzMz5oEAaGwkAAAAAAABEQBEAAAAAAABEQCHNzMzMzPmgQBobCQAAAAAAAERAEQAAAAAAAERAIc3MzMzM+aBAGhsJAAAAAAAAREARAAAAAAAASEAhzczMzMz5oEAaGwkAAAAAAABIQBEAAAAAAIBLQCHNzMzMzPmgQBobCQAAAAAAgEtAEQAAAAAAwFhAIc3MzMzM+aBAIAFCEAoOaG91cnMtcGVyLXdlZWs="></facets-overview>';
        facets_iframe.srcdoc = facets_html;
         facets_iframe.id = "";
         setTimeout(() => {
           facets_iframe.setAttribute('height', facets_iframe.contentWindow.document.body.offsetHeight + 'px')
         }, 1500)
         </script>



<div><b>'eval' split:</b></div><br/>



<iframe id='facets-iframe' width="100%" height="500px"></iframe>
        <script>
        facets_iframe = document.getElementById('facets-iframe');
        facets_html = '<script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"><\/script><link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/master/facets-dist/facets-jupyter.html"><facets-overview proto-input="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"></facets-overview>';
        facets_iframe.srcdoc = facets_html;
         facets_iframe.id = "";
         setTimeout(() => {
           facets_iframe.setAttribute('height', facets_iframe.contentWindow.document.body.offsetHeight + 'px')
         }, 1500)
         </script>


### SchemaGen

The [SchemaGen](https://www.tensorflow.org/tfx/guide/schemagen) component also uses TFDV to generate a schema based on your data statistics. As you've learned previously, a schema defines the expected bounds, types, and properties of the features in your dataset.

`SchemaGen` will take as input the statistics that we generated with `StatisticsGen`, looking at the training split by default.


```python
# Instantiate SchemaGen with the StatisticsGen ingested dataset
schema_gen = tfx.components.SchemaGen(
    statistics=statistics_gen.outputs['statistics'],
    )

# Run the component
context.run(schema_gen)
```




<style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7fd515446fa0</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">3</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">SchemaGen</span><span class="deemphasize"> at 0x7fd4fcdb4280</span></div><table class="attr-table"><tr><td class="attr-name">.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['statistics']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7fd4fcdcba00</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/StatisticsGen/statistics/2)<span class="deemphasize"> at 0x7fd4fcdb4f40</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/StatisticsGen/statistics/2</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7fd4fcdb4d60</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/SchemaGen/schema/3)<span class="deemphasize"> at 0x7fd4fcdb41c0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/SchemaGen/schema/3</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.exec_properties</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['infer_feature_shape']</td><td class = "attrvalue">1</td></tr><tr><td class="attr-name">['exclude_splits']</td><td class = "attrvalue">[]</td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['statistics']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7fd4fcdcba00</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/StatisticsGen/statistics/2)<span class="deemphasize"> at 0x7fd4fcdb4f40</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/StatisticsGen/statistics/2</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7fd4fcdb4d60</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/SchemaGen/schema/3)<span class="deemphasize"> at 0x7fd4fcdb41c0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/SchemaGen/schema/3</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr></table></div>



You can then visualize the generated schema as a table.


```python
# Visualize the schema
context.show(schema_gen.outputs['schema'])
```


<b>Artifact at ./pipeline/SchemaGen/schema/3</b><br/><br/>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Type</th>
      <th>Presence</th>
      <th>Valency</th>
      <th>Domain</th>
    </tr>
    <tr>
      <th>Feature name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'education'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'education'</td>
    </tr>
    <tr>
      <th>'label'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'label'</td>
    </tr>
    <tr>
      <th>'marital-status'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'marital-status'</td>
    </tr>
    <tr>
      <th>'native-country'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'native-country'</td>
    </tr>
    <tr>
      <th>'occupation'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'occupation'</td>
    </tr>
    <tr>
      <th>'race'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'race'</td>
    </tr>
    <tr>
      <th>'relationship'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'relationship'</td>
    </tr>
    <tr>
      <th>'sex'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'sex'</td>
    </tr>
    <tr>
      <th>'workclass'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'workclass'</td>
    </tr>
    <tr>
      <th>'age'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'capital-gain'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'capital-loss'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'education-num'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'fnlwgt'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'hours-per-week'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Values</th>
    </tr>
    <tr>
      <th>Domain</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'education'</th>
      <td>' 10th', ' 11th', ' 12th', ' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' Assoc-acdm', ' Assoc-voc', ' Bachelors', ' Doctorate', ' HS-grad', ' Masters', ' Preschool', ' Prof-school', ' Some-college'</td>
    </tr>
    <tr>
      <th>'label'</th>
      <td>' &lt;=50K', ' &gt;50K'</td>
    </tr>
    <tr>
      <th>'marital-status'</th>
      <td>' Divorced', ' Married-AF-spouse', ' Married-civ-spouse', ' Married-spouse-absent', ' Never-married', ' Separated', ' Widowed'</td>
    </tr>
    <tr>
      <th>'native-country'</th>
      <td>' ?', ' Cambodia', ' Canada', ' China', ' Columbia', ' Cuba', ' Dominican-Republic', ' Ecuador', ' El-Salvador', ' England', ' France', ' Germany', ' Greece', ' Guatemala', ' Haiti', ' Honduras', ' Hong', ' Hungary', ' India', ' Iran', ' Ireland', ' Italy', ' Jamaica', ' Japan', ' Laos', ' Mexico', ' Nicaragua', ' Outlying-US(Guam-USVI-etc)', ' Peru', ' Philippines', ' Poland', ' Portugal', ' Puerto-Rico', ' Scotland', ' South', ' Taiwan', ' Thailand', ' Trinadad&amp;Tobago', ' United-States', ' Vietnam', ' Yugoslavia', ' Holand-Netherlands'</td>
    </tr>
    <tr>
      <th>'occupation'</th>
      <td>' ?', ' Adm-clerical', ' Armed-Forces', ' Craft-repair', ' Exec-managerial', ' Farming-fishing', ' Handlers-cleaners', ' Machine-op-inspct', ' Other-service', ' Priv-house-serv', ' Prof-specialty', ' Protective-serv', ' Sales', ' Tech-support', ' Transport-moving'</td>
    </tr>
    <tr>
      <th>'race'</th>
      <td>' Amer-Indian-Eskimo', ' Asian-Pac-Islander', ' Black', ' Other', ' White'</td>
    </tr>
    <tr>
      <th>'relationship'</th>
      <td>' Husband', ' Not-in-family', ' Other-relative', ' Own-child', ' Unmarried', ' Wife'</td>
    </tr>
    <tr>
      <th>'sex'</th>
      <td>' Female', ' Male'</td>
    </tr>
    <tr>
      <th>'workclass'</th>
      <td>' ?', ' Federal-gov', ' Local-gov', ' Never-worked', ' Private', ' Self-emp-inc', ' Self-emp-not-inc', ' State-gov', ' Without-pay'</td>
    </tr>
  </tbody>
</table>
</div>


Let's now move to the next step in the pipeline and see if there are any anomalies in the data.

### ExampleValidator

The [ExampleValidator](https://www.tensorflow.org/tfx/guide/exampleval) component detects anomalies in your data based on the generated schema from the previous step. Like the previous two components, it also uses TFDV under the hood. 

`ExampleValidator` will take as input the statistics from `StatisticsGen` and the schema from `SchemaGen`. By default, it compares the statistics from the evaluation split to the schema from the training split.


```python
# Instantiate ExampleValidator with the StatisticsGen and SchemaGen ingested data
example_validator = tfx.components.ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema'])

# Run the component.
context.run(example_validator)
```




<style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7fd4fcdcbbe0</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">4</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExampleValidator</span><span class="deemphasize"> at 0x7fd4defc57f0</span></div><table class="attr-table"><tr><td class="attr-name">.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['statistics']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7fd4fcdcba00</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/StatisticsGen/statistics/2)<span class="deemphasize"> at 0x7fd4fcdb4f40</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/StatisticsGen/statistics/2</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7fd4fcdb4d60</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/SchemaGen/schema/3)<span class="deemphasize"> at 0x7fd4fcdb41c0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/SchemaGen/schema/3</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['anomalies']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleAnomalies'</span> (1 artifact)<span class="deemphasize"> at 0x7fd4defc5700</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleAnomalies</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleAnomalies'</span> (uri: ./pipeline/ExampleValidator/anomalies/4)<span class="deemphasize"> at 0x7fd4def83130</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleAnomalies&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/ExampleValidator/anomalies/4</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.exec_properties</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['exclude_splits']</td><td class = "attrvalue">[]</td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['statistics']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7fd4fcdcba00</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/StatisticsGen/statistics/2)<span class="deemphasize"> at 0x7fd4fcdb4f40</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/StatisticsGen/statistics/2</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7fd4fcdb4d60</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/SchemaGen/schema/3)<span class="deemphasize"> at 0x7fd4fcdb41c0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/SchemaGen/schema/3</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['anomalies']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleAnomalies'</span> (1 artifact)<span class="deemphasize"> at 0x7fd4defc5700</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleAnomalies</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleAnomalies'</span> (uri: ./pipeline/ExampleValidator/anomalies/4)<span class="deemphasize"> at 0x7fd4def83130</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleAnomalies&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/ExampleValidator/anomalies/4</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr></table></div>



As with the previous component, you can also visualize the anomalies as a table.


```python
# Visualize the results
context.show(example_validator.outputs['anomalies'])
```


<b>Artifact at ./pipeline/ExampleValidator/anomalies/4</b><br/><br/>



<div><b>'train' split:</b></div><br/>



<h4 style="color:green;">No anomalies found.</h4>



<div><b>'eval' split:</b></div><br/>



<h4 style="color:green;">No anomalies found.</h4>


With no anomalies detected, you can proceed to the next step in the pipeline.

### Transform
The [Transform](https://www.tensorflow.org/tfx/guide/transform) component performs feature engineering for both training and serving datasets. It uses the [TensorFlow Transform](https://www.tensorflow.org/tfx/transform/get_started) library introduced in the first ungraded lab of this week.

`Transform` will take as input the data from `ExampleGen`, the schema from `SchemaGen`, as well as a module containing the preprocessing function.

In this section, you will work on an example of a user-defined Transform code. The pipeline needs to load this as a module so you need to use the magic command `%% writefile` to save the file to disk. Let's first define a few constants that group the data's attributes according to the transforms we will perform later. This file will also be saved locally.


```python
# Set the constants module filename
_census_constants_module_file = 'census_constants.py'
```


```python
%%writefile {_census_constants_module_file}

# Features with string data types that will be converted to indices
CATEGORICAL_FEATURE_KEYS = [
    'education', 'marital-status', 'occupation', 'race', 'relationship', 'workclass', 'sex', 'native-country'
]

# Numerical features that are marked as continuous
NUMERIC_FEATURE_KEYS = ['fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

# Feature that can be grouped into buckets
BUCKET_FEATURE_KEYS = ['age']

# Number of buckets used by tf.transform for encoding each bucket feature.
FEATURE_BUCKET_COUNT = {'age': 4}

# Feature that the model will predict
LABEL_KEY = 'label'

# Utility function for renaming the feature
def transformed_name(key):
    return key + '_xf'
```

    Writing census_constants.py


Next, you will work on the module that contains `preprocessing_fn()`. As you've seen in the previous lab, this function defines how you will transform the raw data into features that your model can train on (i.e. the next step in the pipeline). You will use the [tft module functions](https://www.tensorflow.org/tfx/transform/api_docs/python/tft) to make these transformations. 

*Note: After completing the entire notebook, we encourage you to go back to this section and try different tft functions aside from the ones already provided below. You can also modify the grouping of the feature keys in the constants file if you want. For example, you may want to scale some features to `[0, 1]` while others are scaled to the z-score. This will be good practice for this week's assignment.*


```python
# Set the transform module filename
_census_transform_module_file = 'census_transform.py'
```


```python
%%writefile {_census_transform_module_file}

import tensorflow as tf
import tensorflow_transform as tft

import census_constants

# Unpack the contents of the constants module
_NUMERIC_FEATURE_KEYS = census_constants.NUMERIC_FEATURE_KEYS
_CATEGORICAL_FEATURE_KEYS = census_constants.CATEGORICAL_FEATURE_KEYS
_BUCKET_FEATURE_KEYS = census_constants.BUCKET_FEATURE_KEYS
_FEATURE_BUCKET_COUNT = census_constants.FEATURE_BUCKET_COUNT
_LABEL_KEY = census_constants.LABEL_KEY
_transformed_name = census_constants.transformed_name


# Define the transformations
def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
        inputs: map from feature keys to raw not-yet-transformed features.
    Returns:
        Map from string feature key to transformed feature operations.
    """
    outputs = {}

    # Scale these features to the range [0,1]
    for key in _NUMERIC_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.scale_to_0_1(
            inputs[key])
    
    # Bucketize these features
    for key in _BUCKET_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.bucketize(
            inputs[key], _FEATURE_BUCKET_COUNT[key])

    # Convert strings to indices in a vocabulary
    for key in _CATEGORICAL_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.compute_and_apply_vocabulary(inputs[key])

    # Convert the label strings to an index
    outputs[_transformed_name(_LABEL_KEY)] = tft.compute_and_apply_vocabulary(inputs[_LABEL_KEY])

    return outputs
```

    Writing census_transform.py


You can now pass the training data, schema, and transform module to the `Transform` component. You can ignore the warning messages generated by Apache Beam regarding type hints.


```python
# Ignore TF warning messages
tf.get_logger().setLevel('ERROR')

# Instantiate the Transform component
transform = tfx.components.Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file=os.path.abspath(_census_transform_module_file))

# Run the component
context.run(transform)
```

    WARNING:root:This output type hint will be ignored and not used for type-checking purposes. Typically, output type hints for a PTransform are single (or nested) types wrapped by a PCollection, PDone, or None. Got: Tuple[Dict[str, Union[NoneType, _Dataset]], Union[Dict[str, Dict[str, PCollection]], NoneType], int] instead.
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary_1/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary_2/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary_3/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary_4/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary_5/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary_6/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary_7/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary_8/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary_1/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary_2/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary_3/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary_4/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary_5/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary_6/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary_7/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:absl:Tables initialized inside a tf.function  will be re-initialized on every invocation of the function. This  re-initialization can have significant impact on performance. Consider lifting  them out of the graph context using  `tf.init_scope`.: compute_and_apply_vocabulary_8/apply_vocab/text_file_init/InitializeTableFromTextFileV2
    WARNING:root:This output type hint will be ignored and not used for type-checking purposes. Typically, output type hints for a PTransform are single (or nested) types wrapped by a PCollection, PDone, or None. Got: Tuple[Dict[str, Union[NoneType, _Dataset]], Union[Dict[str, Dict[str, PCollection]], NoneType], int] instead.
    WARNING:root:Make sure that locally built Python SDK docker image has Python 3.8 interpreter.





<style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object expanded"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">ExecutionResult</span><span class="deemphasize"> at 0x7fd4fc7f2970</span></div><table class="attr-table"><tr><td class="attr-name">.execution_id</td><td class = "attrvalue">5</td></tr><tr><td class="attr-name">.component</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Transform</span><span class="deemphasize"> at 0x7fd4fc795d00</span></div><table class="attr-table"><tr><td class="attr-name">.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7fd394736250</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/CsvExampleGen/examples/1)<span class="deemphasize"> at 0x7fd394736610</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/CsvExampleGen/examples/1</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7fd4fcdb4d60</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/SchemaGen/schema/3)<span class="deemphasize"> at 0x7fd4fcdb41c0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/SchemaGen/schema/3</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['transform_graph']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'TransformGraph'</span> (1 artifact)<span class="deemphasize"> at 0x7fd4fc795970</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">TransformGraph</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'TransformGraph'</span> (uri: ./pipeline/Transform/transform_graph/5)<span class="deemphasize"> at 0x7fd4fc795a30</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.TransformGraph&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/transform_graph/5</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['transformed_examples']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7fd4fc795fa0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/Transform/transformed_examples/5)<span class="deemphasize"> at 0x7fd4fc795a90</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/transformed_examples/5</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['updated_analyzer_cache']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'TransformCache'</span> (1 artifact)<span class="deemphasize"> at 0x7fd4fc795ca0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">TransformCache</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'TransformCache'</span> (uri: ./pipeline/Transform/updated_analyzer_cache/5)<span class="deemphasize"> at 0x7fd4fc795820</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.TransformCache&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/updated_analyzer_cache/5</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['pre_transform_schema']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7fd4fc795d60</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/Transform/pre_transform_schema/5)<span class="deemphasize"> at 0x7fd4fc795cd0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/pre_transform_schema/5</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['pre_transform_stats']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7fd4fc795c70</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/Transform/pre_transform_stats/5)<span class="deemphasize"> at 0x7fd4fc795c10</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/pre_transform_stats/5</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue"></td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['post_transform_schema']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7fd4fc7958e0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/Transform/post_transform_schema/5)<span class="deemphasize"> at 0x7fd4fc7957c0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/post_transform_schema/5</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['post_transform_stats']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7fd4fc795d90</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/Transform/post_transform_stats/5)<span class="deemphasize"> at 0x7fd4fc795af0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/post_transform_stats/5</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue"></td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['post_transform_anomalies']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleAnomalies'</span> (1 artifact)<span class="deemphasize"> at 0x7fd4def83700</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleAnomalies</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleAnomalies'</span> (uri: ./pipeline/Transform/post_transform_anomalies/5)<span class="deemphasize"> at 0x7fd4fc795b20</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleAnomalies&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/post_transform_anomalies/5</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue"></td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.exec_properties</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['module_file']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['preprocessing_fn']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['stats_options_updater_fn']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['force_tf_compat_v1']</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">['custom_config']</td><td class = "attrvalue">null</td></tr><tr><td class="attr-name">['splits_config']</td><td class = "attrvalue">None</td></tr><tr><td class="attr-name">['disable_statistics']</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">['module_path']</td><td class = "attrvalue">census_transform@./pipeline/_wheels/tfx_user_code_Transform-0.0+54e97b1f397e91392efd7550f0e15993fbc6b1b005c95821d088b5802c0e3f96-py3-none-any.whl</td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">.component.inputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['examples']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7fd394736250</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/CsvExampleGen/examples/1)<span class="deemphasize"> at 0x7fd394736610</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/CsvExampleGen/examples/1</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['schema']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7fd4fcdb4d60</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/SchemaGen/schema/3)<span class="deemphasize"> at 0x7fd4fcdb41c0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/SchemaGen/schema/3</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr><tr><td class="attr-name">.component.outputs</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">['transform_graph']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'TransformGraph'</span> (1 artifact)<span class="deemphasize"> at 0x7fd4fc795970</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">TransformGraph</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'TransformGraph'</span> (uri: ./pipeline/Transform/transform_graph/5)<span class="deemphasize"> at 0x7fd4fc795a30</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.TransformGraph&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/transform_graph/5</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['transformed_examples']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Examples'</span> (1 artifact)<span class="deemphasize"> at 0x7fd4fc795fa0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Examples</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Examples'</span> (uri: ./pipeline/Transform/transformed_examples/5)<span class="deemphasize"> at 0x7fd4fc795a90</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Examples&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/transformed_examples/5</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue">[&quot;train&quot;, &quot;eval&quot;]</td></tr><tr><td class="attr-name">.version</td><td class = "attrvalue">0</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['updated_analyzer_cache']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'TransformCache'</span> (1 artifact)<span class="deemphasize"> at 0x7fd4fc795ca0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">TransformCache</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'TransformCache'</span> (uri: ./pipeline/Transform/updated_analyzer_cache/5)<span class="deemphasize"> at 0x7fd4fc795820</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.TransformCache&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/updated_analyzer_cache/5</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['pre_transform_schema']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7fd4fc795d60</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/Transform/pre_transform_schema/5)<span class="deemphasize"> at 0x7fd4fc795cd0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/pre_transform_schema/5</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['pre_transform_stats']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7fd4fc795c70</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/Transform/pre_transform_stats/5)<span class="deemphasize"> at 0x7fd4fc795c10</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/pre_transform_stats/5</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue"></td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['post_transform_schema']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'Schema'</span> (1 artifact)<span class="deemphasize"> at 0x7fd4fc7958e0</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">Schema</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'Schema'</span> (uri: ./pipeline/Transform/post_transform_schema/5)<span class="deemphasize"> at 0x7fd4fc7957c0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.Schema&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/post_transform_schema/5</td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['post_transform_stats']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleStatistics'</span> (1 artifact)<span class="deemphasize"> at 0x7fd4fc795d90</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleStatistics</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleStatistics'</span> (uri: ./pipeline/Transform/post_transform_stats/5)<span class="deemphasize"> at 0x7fd4fc795af0</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleStatistics&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/post_transform_stats/5</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue"></td></tr></table></div></td></tr></table></td></tr></table></div></td></tr><tr><td class="attr-name">['post_transform_anomalies']</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Channel</span> of type <span class="class-name">'ExampleAnomalies'</span> (1 artifact)<span class="deemphasize"> at 0x7fd4def83700</span></div><table class="attr-table"><tr><td class="attr-name">.type_name</td><td class = "attrvalue">ExampleAnomalies</td></tr><tr><td class="attr-name">._artifacts</td><td class = "attrvalue"><table class="attr-table"><tr><td class="attr-name">[0]</td><td class = "attrvalue"><style>
.tfx-object.expanded {
  padding: 4px 8px 4px 8px;
  background: white;
  border: 1px solid #bbbbbb;
  box-shadow: 4px 4px 2px rgba(0,0,0,0.05);
}
.tfx-object, .tfx-object * {
  font-size: 11pt;
}
.tfx-object > .title {
  cursor: pointer;
}
.tfx-object .expansion-marker {
  color: #999999;
}
.tfx-object.expanded > .title > .expansion-marker:before {
  content: '▼';
}
.tfx-object.collapsed > .title > .expansion-marker:before {
  content: '▶';
}
.tfx-object .class-name {
  font-weight: bold;
}
.tfx-object .deemphasize {
  opacity: 0.5;
}
.tfx-object.collapsed > table.attr-table {
  display: none;
}
.tfx-object.expanded > table.attr-table {
  display: block;
}
.tfx-object table.attr-table {
  border: 2px solid white;
  margin-top: 5px;
}
.tfx-object table.attr-table td.attr-name {
  vertical-align: top;
  font-weight: bold;
}
.tfx-object table.attr-table td.attrvalue {
  text-align: left;
}
</style>
<script>
function toggleTfxObject(element) {
  var objElement = element.parentElement;
  if (objElement.classList.contains('collapsed')) {
    objElement.classList.remove('collapsed');
    objElement.classList.add('expanded');
  } else {
    objElement.classList.add('collapsed');
    objElement.classList.remove('expanded');
  }
}
</script>
<div class="tfx-object collapsed"><div class = "title" onclick="toggleTfxObject(this)"><span class="expansion-marker"></span><span class="class-name">Artifact</span> of type <span class="class-name">'ExampleAnomalies'</span> (uri: ./pipeline/Transform/post_transform_anomalies/5)<span class="deemphasize"> at 0x7fd4fc795b20</span></div><table class="attr-table"><tr><td class="attr-name">.type</td><td class = "attrvalue">&lt;class &#x27;tfx.types.standard_artifacts.ExampleAnomalies&#x27;&gt;</td></tr><tr><td class="attr-name">.uri</td><td class = "attrvalue">./pipeline/Transform/post_transform_anomalies/5</td></tr><tr><td class="attr-name">.span</td><td class = "attrvalue">0</td></tr><tr><td class="attr-name">.split_names</td><td class = "attrvalue"></td></tr></table></div></td></tr></table></td></tr></table></div></td></tr></table></td></tr></table></div>



Let's examine the output artifacts of `Transform` (i.e. `.component.outputs` from the output cell above). This component produces several outputs:

* `transform_graph` is the graph that can perform the preprocessing operations. This graph will be included during training and serving to ensure consistent transformations of incoming data.
* `transformed_examples` points to the preprocessed training and evaluation data.
* `updated_analyzer_cache` are stored calculations from previous runs.

Take a peek at the `transform_graph` artifact.  It points to a directory containing three subdirectories.


```python
# Get the uri of the transform graph
transform_graph_uri = transform.outputs['transform_graph'].get()[0].uri

# List the subdirectories under the uri
os.listdir(transform_graph_uri)
```




    ['metadata', 'transformed_metadata', 'transform_fn']



* The `metadata` subdirectory contains the schema of the original data.
* The `transformed_metadata` subdirectory contains the schema of the preprocessed data. 
* The `transform_fn` subdirectory contains the actual preprocessing graph. 

You can also take a look at the first three transformed examples using the helper function defined earlier.


```python
# Get the URI of the output artifact representing the transformed examples
train_uri = os.path.join(transform.outputs['transformed_examples'].get()[0].uri, 'Split-train')

# Get the list of files in this directory (all compressed TFRecord files)
tfrecord_filenames = [os.path.join(train_uri, name)
                      for name in os.listdir(train_uri)]

# Create a `TFRecordDataset` to read these files
transformed_dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type="GZIP")
```


```python
# Get 3 records from the dataset
sample_records_xf = get_records(transformed_dataset, 3)

# Print the output
pp.pprint(sample_records_xf)
```

    [{'features': {'feature': {'age_xf': {'int64List': {'value': ['2']}},
                               'capital-gain_xf': {'floatList': {'value': [0.021740217]}},
                               'capital-loss_xf': {'floatList': {'value': [0.0]}},
                               'education-num_xf': {'floatList': {'value': [0.8]}},
                               'education_xf': {'int64List': {'value': ['2']}},
                               'fnlwgt_xf': {'floatList': {'value': [0.044301897]}},
                               'hours-per-week_xf': {'floatList': {'value': [0.39795917]}},
                               'label_xf': {'int64List': {'value': ['0']}},
                               'marital-status_xf': {'int64List': {'value': ['1']}},
                               'native-country_xf': {'int64List': {'value': ['0']}},
                               'occupation_xf': {'int64List': {'value': ['3']}},
                               'race_xf': {'int64List': {'value': ['0']}},
                               'relationship_xf': {'int64List': {'value': ['1']}},
                               'sex_xf': {'int64List': {'value': ['0']}},
                               'workclass_xf': {'int64List': {'value': ['4']}}}}},
     {'features': {'feature': {'age_xf': {'int64List': {'value': ['3']}},
                               'capital-gain_xf': {'floatList': {'value': [0.0]}},
                               'capital-loss_xf': {'floatList': {'value': [0.0]}},
                               'education-num_xf': {'floatList': {'value': [0.8]}},
                               'education_xf': {'int64List': {'value': ['2']}},
                               'fnlwgt_xf': {'floatList': {'value': [0.048237596]}},
                               'hours-per-week_xf': {'floatList': {'value': [0.12244898]}},
                               'label_xf': {'int64List': {'value': ['0']}},
                               'marital-status_xf': {'int64List': {'value': ['0']}},
                               'native-country_xf': {'int64List': {'value': ['0']}},
                               'occupation_xf': {'int64List': {'value': ['0']}},
                               'race_xf': {'int64List': {'value': ['0']}},
                               'relationship_xf': {'int64List': {'value': ['0']}},
                               'sex_xf': {'int64List': {'value': ['0']}},
                               'workclass_xf': {'int64List': {'value': ['1']}}}}},
     {'features': {'feature': {'age_xf': {'int64List': {'value': ['2']}},
                               'capital-gain_xf': {'floatList': {'value': [0.0]}},
                               'capital-loss_xf': {'floatList': {'value': [0.0]}},
                               'education-num_xf': {'floatList': {'value': [0.53333336]}},
                               'education_xf': {'int64List': {'value': ['0']}},
                               'fnlwgt_xf': {'floatList': {'value': [0.13811344]}},
                               'hours-per-week_xf': {'floatList': {'value': [0.39795917]}},
                               'label_xf': {'int64List': {'value': ['0']}},
                               'marital-status_xf': {'int64List': {'value': ['2']}},
                               'native-country_xf': {'int64List': {'value': ['0']}},
                               'occupation_xf': {'int64List': {'value': ['9']}},
                               'race_xf': {'int64List': {'value': ['0']}},
                               'relationship_xf': {'int64List': {'value': ['1']}},
                               'sex_xf': {'int64List': {'value': ['0']}},
                               'workclass_xf': {'int64List': {'value': ['0']}}}}}]


**Congratulations!** You have now executed all the components in our pipeline. You will get hands-on practice as well with training and model evaluation in future courses but for now, we encourage you to try exploring the different components we just discussed. As mentioned earlier, a useful exercise for the upcoming assignment is to be familiar with using different `tft` functions in your transform module. Try exploring the [documentation](https://www.tensorflow.org/tfx/transform/api_docs/python/tft) and see what other functions you can use in the transform module. You can also do the optional challenge below for more practice.

**Optional Challenge:** Using this notebook as reference, load the [Seoul Bike Sharing Demand Dataset](https://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand) and run it through the five stages of the pipeline discussed here. You will first go through the data ingestion and validation components then finally, you will study the dataset's features and transform it to a format that a model can consume. Once you're done, you can visit this [Discourse topic](https://community.deeplearning.ai/t/bike-sharing-dataset-in-c2-w2-lab-2-feature-engineering-pipeline/38979) where one of your mentors, Fabio, has shared his solution. Feel free to discuss and share your solution as well!
