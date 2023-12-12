---
lab:
    title: 'Explore classification with Azure Machine Learning Designer'
---

# Explore classification with Azure Machine Learning Designer

> **Note**
> To complete this lab, you will need an [Azure subscription](https://azure.microsoft.com/free?azure-portal=true) in which you have administrative access.

## Create an Azure Machine Learning workspace  

1. Sign into the [Azure portal](https://portal.azure.com?azure-portal=true) using your Microsoft credentials.

1. Select **+ Create a resource**, search for *Machine Learning*, and create a new **Azure Machine Learning** resource with an *Azure Machine Learning* plan. Use the following settings:
    - **Subscription**: *Your Azure subscription*.
    - **Resource group**: *Create or select a resource group*.
    - **Workspace name**: *Enter a unique name for your workspace*.
    - **Region**: *Select the closest geographical region*.
    - **Storage account**: *Note the default new storage account that will be created for your workspace*.
    - **Key vault**: *Note the default new key vault that will be created for your workspace*.
    - **Application insights**: *Note the default new application insights resource that will be created for your workspace*.
    - **Container registry**: None (*one will be created automatically the first time you deploy a model to a container*)

1. Select **Review + create**, then select **Create**. Wait for your workspace to be created (it can take a few minutes), and then go to the deployed resource.

1. Select **Launch studio** (or open a new browser tab and navigate to [https://ml.azure.com](https://ml.azure.com?azure-portal=true), and sign into Azure Machine Learning studio using your Microsoft account).

1. In Azure Machine Learning studio, you should see your newly created workspace. If that is not the case, select your Azure directory in the left-hand menu. Then from the new left-hand menu select **Workspaces**, where all the workspaces associated to your directory are listed, and select the one you created for this exercise.

> **Note**
> This module is one of many that make use of an Azure Machine Learning workspace, including the other modules in the [Microsoft Azure AI Fundamentals: Explore visual tools for machine learning](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/) learning path. If you are using your own Azure subscription, you may consider creating the workspace once and reusing it in other modules. Your Azure subscription will be charged a small amount for data storage as long as the Azure Machine Learning workspace exists in your subscription, so we recommend you delete the Azure Machine Learning workspace when it is no longer required.

## Create compute

1. In [Azure Machine Learning studio](https://ml.azure.com?azure-portal=true), select the **&#8801;** icon (a menu icon that looks like a stack of three lines) at the top left to view the various pages in the interface (you may need to maximize the size of your screen). You can use these pages in the left hand pane to manage the resources in your workspace. Select the **Compute** page (under **Manage**).

1. On the **Compute** page, select the **Compute clusters** tab, and add a new compute cluster with the following settings. You'll use this to train a machine learning model:
    - **Location**: *Select the same as your workspace. If that location is not listed, choose the one closest to you*.
    - **Virtual machine tier**: Dedicated
    - **Virtual machine type**: CPU
    - **Virtual machine size**:
        - Choose **Select from all options**
        - Search for and select **Standard_DS11_v2**
    - Select **Next**
    - **Compute name**: *enter a unique name*.
    - **Minimum number of nodes**: 0
    - **Maximum number of nodes**: 2
    - **Idle seconds before scale down**: 120
    - **Enable SSH access**: Clear
    - Select **Create**

> **Note**
> Compute instances and clusters are based on standard Azure virtual machine images. For this module, the *Standard_DS11_v2* image is recommended to achieve the optimal balance of cost and performance. If your subscription has a quota that does not include this image, choose an alternative image; but bear in mind that a larger image may incur higher cost and a smaller image may not be sufficient to complete the tasks. Alternatively, ask your Azure administrator to extend your quota.

The compute cluster will take some time to be created. You can move onto the next step while you wait.


## Create a dataset

1. In [Azure Machine Learning studio](https://ml.azure.com?azure-portal=true), expand the left pane by selecting the menu icon at the top left of the screen. Select the **Data** page (under **Assets**). The Data page contains specific data files or tables that you plan to work with in Azure ML. You can create datasets from this page as well.

1. On the **Data** page, under the **Data assets** tab, select **+ Create**. Then configure a data asset with the following settings:
    - **Data type**:
        - **Name**: diabetes-data
        - **Description**: Diabetes data
        - **Dataset type**: Tabular
    - **Data source**: From web files
    - **Web URL**:
        - **Web URL**: https://aka.ms/diabetes-data
        - **Skip data validation**: *do not select*
    - **Settings**:
        - **File format**: Delimited
        - **Delimiter**: Comma
        - **Encoding**: UTF-8
        - **Column headers**: Only first file has headers
        - **Skip rows**: None
        - **Dataset contains multi-line data**: *do not select*
    - **Schema**:
        - Include all columns other than **Path**
        - Review the automatically detected types
    - **Review**
        - Select **Create**

1. After the dataset has been created, open it and view the **Explore** page to see a sample of the data. This data represents details from patients who have been tested for diabetes.

## Create a pipeline in Designer and load data to canvas

To get started with Azure Machine Learning designer, first you must create a pipeline and add the dataset you want to work with.

1. In [Azure Machine Learning studio](https://ml.azure.com?azure-portal=true), on the left pane select the **Designer** item (under **Authoring**), and then select **+** to create a new pipeline.

1. Change the draft name from **Pipeline-Created-on-*date*** to **Diabetes Training**.

1. Then in the project, next to the pipeline name on the left, select the arrows icon to expand the panel if it is not already expanded. The panel should open by default to the **Asset library** pane, indicated by the books icon at the top of the panel. Note that there is a search bar to locate assets. Notice two buttons, **Data** and **Component**.

    ![Screenshot of location of designer asset library, search bar, and data icon.](media/create-classification-model/designer-asset-library-data.png)

1. Select **Data**. Search for and place the **diabetes-data** dataset onto the canvas.

1. Right-click (Ctrl+click on a Mac) the **diabetes-data** dataset on the canvas, and select **Preview data**.

1. Review the schema of the data in the *Profile* tab, noting that you can see the distributions of the various columns as histograms.

1. Scroll down and select the column heading for the **Diabetic** column, and note that it contains two values **0** and **1**. These values represent the two possible classes for the *label* that your model will predict, with a value of **0** meaning that the patient does not have diabetes, and a value of **1** meaning that the patient is diabetic.

1. Scroll back up and review the other columns, which represent the *features* that will be used to predict the label. Note that most of these columns are numeric, but each feature is on its own scale. For example, **Age** values range from 21 to 77, while **DiabetesPedigree** values range from 0.078 to 2.3016. When training a machine learning model, it is sometimes possible for larger values to dominate the resulting predictive function, reducing the influence of features that on a smaller scale. Typically, data scientists mitigate this possible bias by *normalizing* the numeric columns so they're on the similar scales.

1. Close the **DataOutput** tab so that you can see the dataset on the canvas like this:

    ![Screenshot of the diabetes-data dataset on the designer canvas.](media/create-classification-model/diabetes-data.png)

## Add transformations

Before you can train a model, you typically need to apply some pre-processing transformations to the data.

1. In the **Asset library** pane on the left, select **Component**, which contains a wide range of modules you can use for data transformation and model training. You can also use the search bar to quickly locate modules.

    ![Screenshot of location of designer asset library, search bar, and components icon.](media/create-classification-model/designer-asset-library-components.png)

1. Find the **Select Columns in Dataset** module and place it on the canvas below the **diabetes-data** dataset. Then connect the output from the bottom of the **diabetes-data** dataset to the input at the top of the **Select Columns in Dataset** module.

1. Double click on the **Select Columns in Dataset** module to access a settings pane on the right. Select **Edit column**. Then in the **Select columns** window, select **By name** and **Add all** the columns. Then remove **PatientID** and click **Save**.

1. Find the **Normalize Data** module and place it on the canvas below the **Select Columns in Dataset** module. Then connect the output from the bottom of the **Select Columns in Dataset** module to the input at the top of the **Normalize Data** module, like this:

    ![Screenshot of a pipeline with the dataset connected to select columns and Normalize Data module.](media/create-classification-model/dataset-normalize.png)

1. Double-click the **Normalize Data** module to view its settings, noting that it requires you to specify the transformation method and the columns to be transformed.

1. Set the *Transformation method* to **MinMax** and the *Use 0 for constant columns when checked* to **True**. Edit the columns to transform with **Edit columns**. Select columns **With Rules** and copy and paste the following list under include column names:  

```
Pregnancies, PlasmaGlucose, DiastolicBloodPressure, TricepsThickness, SerumInsulin, BMI, DiabetesPedigree, Age
```
![Screenshot of the columns selected for normalization.](media/create-classification-model/normalize-data.png)

Click **Save** and close the selection box.

The data transformation is normalizing the numeric columns to put them on the same scale, which should help prevent columns with large values from dominating model training. You'd usually apply a whole bunch of pre-processing transformations like this to prepare your data for training, but we'll keep things simple in this exercise.

## Run the pipeline

To apply your data transformations, you need to run the pipeline as an experiment.

1. Select **Configure & Submit** at the top of the page to open the **Set up pipeline job** dialogue.

1. On the **Basics** page select **Create new** and set the name of the experiment to **mslearn-diabetes-training** then select **Next** .

1. On the **Inputs & outputs** page select **Next** without making any changes.

1. On the **Runtime settings** page an error appears as you don´t have a default compute to run the pipeline. In the **Select compute type** drop-down select *Compute cluster* and in the **Select Azure ML compute cluster** drop-down select your recently created compute cluster.

1. Select **Review + Submit** to review the pipeline job and then select **Submit** to run the training pipeline.

1. Wait a few minutes for the run to finish. You can check the status of the job by selecting **Jobs** under the **Assets**. From there, select the **mslearn-diabetes-training** experiment and then the **Diabetes Training** job.

## View the transformed data

When the run has completed, the dataset is now prepared for model training.

1. Right-click (Ctrl+click on a Mac) the **Normalize Data** module on the canvas, and select **Preview data**. Select **Transformed dataset**.

1. View the data, noting that the numeric columns you selected have been normalized to a common scale.

1. Close the normalized data result visualization. Return to the previous tab.

After you've used data transformations to prepare the data, you can use it to train a machine learning model.

## Add training modules

It's common practice to train the model using a subset of the data, while holding back some data with which to test the trained model. This enables you to compare the labels that the model predicts with the actual known labels in the original dataset.

In this exercise, you're going to work through steps to extend the **Diabetes Training** pipeline as shown here:

![Screenshot of how to split data, then train with logistic regression and score.](media/create-classification-model/train-score-pipeline.png)

Follow the steps below, using the image above for reference as you add and configure the required modules.

1. Return to the **Designer** page and select the **Diabetes Training** pipeline.

1. In the **Asset library** pane on the left, in **Component**, search for and place a **Split Data** module onto the canvas under the **Normalize Data** module. Then connect the *Transformed Dataset* (left) output of the **Normalize Data** module to the input of the **Split Data** module.

    >**Tip**
    > Use the search bar to quickly locate modules.

1. Select the **Split Data** module, and configure its settings as follows:
    - **Splitting mode**: Split Rows
    - **Fraction of rows in the first output dataset**: 0.7
    - **Randomized split**: True
    - **Random seed**: 123
    - **Stratified split**: False

1. In the **Asset library**, search for and place a **Train Model** module to the canvas, under the **Split Data** module. Then connect the *Results dataset1* (left) output of the **Split Data** module to the *Dataset* (right) input of the **Train Model** module.

1. The model we're training will predict the **Diabetic** value, so select the **Train Model** module and modify its settings to set the **Label column** to **Diabetic**.

    The **Diabetic** label the model will predict is a class (0 or 1), so we need to train the model using a *classification* algorithm. Specifically, there are two possible classes, so we need a *binary classification* algorithm.

1. In the **Asset library**, search for and place a **Two-Class Logistic Regression** module to the canvas, to the left of the **Split Data** module and above the **Train Model** module. Then connect its output to the *Untrained model* (left) input of the **Train Model** module.

   To test the trained model, we need to use it to *score* the validation dataset we held back when we split the original data - in other words, predict labels for the features in the validation dataset.

1. In the **Asset library**, search for and place a **Score Model** module to the canvas, below the **Train Model** module. Then connect the output of the **Train Model** module to the *Trained model* (left) input of the **Score Model** module; and connect the *Results dataset2* (right) output of the **Split Data** module to the *Dataset* (right) input of the **Score Model** module.

## Run the training pipeline

Now you're ready to run the training pipeline and train the model.

1. Select **Configure & Submit**, and run the pipeline using the existing experiment named **mslearn-diabetes-training**.

1. Wait for the experiment run to finish. This may take 5 minutes or more.

1. Check the status of the job by selecting **Jobs** under the **Assets**. From there, select the **mslearn-diabetes-training** experiment and then select the latest **Diabetes Training** job.

1. On the new tab, right-click (Ctrl+click on a Mac) the **Score Model** module on the canvas, select **Preview data** and then select **Scored dataset** to view the results.

1. Scroll to the right, and note that next to the **Diabetic** column (which contains the known true values of the label) there is a new column named **Scored Labels**, which contains the predicted label values, and a **Scored Probabilities** column containing a probability value between 0 and 1. This indicates the probability of a *positive* prediction, so probabilities greater than 0.5 result in a predicted label of ***1*** (diabetic), while probabilities between 0 and 0.5 result in a predicted label of ***0*** (not diabetic).

1. Close the **Scored_dataset** tab.

The model is predicting values for the **Diabetic** label, but how reliable are its predictions? To assess that, you need to evaluate the model.

The validation data you held back and used to score the model includes the known values for the label. So to validate the model, you can compare the true values for the label to the label values that were predicted when you scored the validation dataset. Based on this comparison, you can calculate various metrics that describe how well the model performs.

## Add an Evaluate Model module

1. Return to **Designer** and open the **Diabetes Training** pipeline you created.

1. In the **Asset library**, search for and place an **Evaluate Model** module to the canvas, under the **Score Model** module, and connect the output of the **Score Model** module to the *Scored dataset* (left) input of the **Evaluate Model** module.

1. Ensure your pipeline looks like this:

    ![Screenshot of the Evaluate Model module added to Score Model module.](media/create-classification-model/evaluate-pipeline.png)

1. Select **Configure & Submit**, and run the pipeline using the existing experiment named **mslearn-diabetes-training**.

1. Wait for the experiment run to finish.

1. Check the status of the job by selecting **Jobs** under the **Assets**. From there, select the **mslearn-diabetes-training** experiment and then select the latest **Diabetes Training** job.

1. On the new tab, right-click (Ctrl+click on a Mac) the **Evaluate Model** module on the canvas, select **Preview data** then select **Evaluation results** to view the performance metrics. These metrics can help data scientists assess how well the model predicts based on the validation data.

1. Scroll down to view the *confusion matrix* for the model. Observe the predicted and actual value counts for each possible class. 

1. Review the metrics to the left of the confusion matrix, which include:
    - **Accuracy**: In other words, what proportion of diabetes predictions did the model get right?
    - **Precision**: In other words, out of all the patients that *the model predicted* as having diabetes, the percentage of time the model is correct. 
    - **Recall**:  In other words, out of all the patients *who actually have* diabetes, how many diabetic cases did the model identify correctly?
    - **F1 Score**

1. Use the **Threshold** slider located above the list of metrics. Try moving the threshold slider and observe the effect on the confusion matrix. If you move it all the way to the left (0), the Recall metric becomes 1, and if you move it all the way to the right (1), the Recall metric becomes 0.

1. Look above the Threshold slider at the **ROC curve** and **AUC** metric listed with the other metrics below. To get an idea of how this area represents the performance of the model, imagine a straight diagonal line from the bottom left to the top right of the ROC chart. This represents the expected performance if you just guessed or flipped a coin for each patient - you could expect to get around half of them right, and half of them wrong, so the area under the diagonal line represents an AUC of 0.5. If the AUC for your model is higher than this for a binary classification model, then the model performs better than a random guess.

1. Close the **Evaluation_results** tab.

The performance of this model isn't all that great, partly because we performed only minimal feature engineering and pre-processing. You could try a different classification algorithm, such as **Two-Class Decision Forest**, and compare the results. You can connect the outputs of the **Split Data** module to multiple **Train Model** and **Score Model** modules, and you can connect a second **Score Model** module to the **Evaluate Model** module to see a side-by-side comparison. The point of the exercise is simply to introduce you to classification and the Azure Machine Learning designer interface, not to train a perfect model!


> To delete your workspace:
>
> 1. In the [Azure portal](https://portal.azure.com?azure-portal=true), in the **Resource groups** page, open the resource group you specified when creating your Azure Machine Learning workspace.
> 1. Click **Delete resource group**, type the resource group name to confirm you want to delete it, and select **Delete**.
