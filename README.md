# Zero Shot Image Tagging
Tagging unseen images with unseen tags


## ABSTRACT

[Splash](https://github.com/nitesh-dubey/Splash-Image-Search) is an Image Search Engine, where the user can search for images by typing tags in the search bar of the website, and can also upload their own images to the website.
Since a crucial part of this image search engine is image retrieval. So each image needs to contain some tags/words related to it, which can describe the contents of the image, and will help in retrieving the image when the user queries it.

Therefore, when the users upload their own images to the website, those images need to be tagged as well. Since it isn’t a good user experience to ask the users to manually tag every image they upload, an automated Image Tagging method is needed. This project describes a deep learning based approach for automated image tagging.


## MOTIVATION

In social media and photo sharing websites like Flickr, Unsplash, pixabay, thte number of tags in an image keep growing as the users keep contributing more tags over time. By the time we train our image tagging model and launch it, new tags will be added to the website. So it's not possible to retrain our model on new tags every time.

So we need to develop a system, which can assign multiple tags to a query image, and can also handle new unseen tags without retraining the model on those tags.


## CONVENTIONAL APPROACHES

Conventional Image Tagging method is basically a multiclass Image Classification problem which is trained on a fixed number of tags, so an input image is classified only into the tags on which the model is trained. If we want to add any new unseen tag to our vocabulary, we have to retrain the model.
Nearest Neighbour Method is also a conventional image tagging method, where we find features of each image and attach tags to it. When we get a query image, we find nearest visually similar images, and then use neighbour’s image tags to tag the query image. But this is computationally expensive and requires O(n) during querying.


## PROBLEM DEFINITION

We need to develop a system which could assign both seen tags and unseen tags to an input query image, without retraining the model for new tags. Where seen tags are the tags which are seen during training, and unseen tags are the new tags introduced later, which aren’t seen during training stage

Consider that model is trained on T number of tags. When a user uses this model for image tagging, the model will assign and rank those T tags on this image.
Over time, imagine that t more tags get added to the system, so now the system has T  seen tags and  t unseen tags. Now, when the model is used for tagging, it needs to rank (T + t) tags on the input query image.


## PROPOSED SYSTEM

We can use linguistic similarities to assign tags to an image without providing any training examples for those tags. This problem is called zero shot learning - the ability to assign labels without any prior training on those labels. 

This problem can be broken down into 4 sub-problems.
Generate an image-feature space for containing relevant features for the image
Generate a word-feature-space that captures semantic and syntactic relations of the words (tags)
Find a mapping that projects the image from the image-feature-space to the word-feature-space. This projection in the word-feature-space is representative of the input image, and is called Principle Direction
Then use the Principle direction of the input image to rank all the candidate tags on this image.
- #### SUB-PROBLEM 1 : GENERATING THE IMAGE-FEATURE-SPACE

This problem is well-documented and can be accomplished by training a Convolutional Neural Network. 
CNNs classify images by first learning complex features to describe the image before passing these features to a standard classifier to perform the desired classification. By removing the classification component, called the fully-connected layer, we can extract just the relevant image’s features.

Since we want generality in our algorithm, so we can tag as many different types of images as possible, I’ve used a CNN architecture, InceptionV3, that was trained on images from the ImageNet competition.


- #### SUB-PROBLEM 2 : GENERATING THE WORD-FEATURE-SPACE

This is also a well documented problem. I’ve used Glove-200d word embeddings to accomplish this, which generates a 200d word vector for each input tag / word. Our tags will just be a small subspace of the entire word vector space.


- #### SUB-PROBLEM 3 : FINDING PRINCIPLE DIRECTION

The key idea is that a list of relevant tags for an image lie along a single direction in word feature space. This direction is called Principle Direction. The Most relevant (+ve) tags rank higher on this principle direction and the lesser relevant tags (-ve) rank lower on the principle direction.
So the problem boils down to finding principle direction in the word feature space for each image and then projecting all candidate tags (both seen and unseen) tags upon it. The principle direction will rank all those candidate tags. Then the top few tags will describe the image perfectly.








## LOSS FUNCTION 

We know that for an image m, there exists a Principle direction **f(xm)** along which the **relevant tags / positive tags ( p )** rank ahead of **irrelevant tags / negative tags ( n )**. This generalises to seen as well as unseen tags.
Let np be the number of positive tags and nn be the number of negative tags

So, we can deduce that the inner product of principle direction and +ve tags will be greater than the inner product of principle direction and -ve tags.
i.e. **<f(xm), p> > <f(xm), n>**

Let,  **loss(p, n) = <f(xm), n> - <f(xm), n>**
So, total loss is :

**L = 𝝨For all images((1/(np * nn)) * 𝝨For all p (𝝨For all n (log(1 + exp{loss(p,n)}))))**

On minimizing this loss function, we get Optimal weights of the neural network which will give their best prediction of Principle direction f(xm) for any image xm.




## DATA PREPARATION TRAINING (Implementation details)

- I had scraped around 6.5k images from pixabay.com along with their +ve and -ve tags from pixabay.com.
- Then I cleaned and processed the text and image data and divided it into a train and test set.
- Second last layer of InceptionV3 was used to get the image features. The dimension of image feature vector is (1, 2048)
- Glove 200d word embeddings were used to get the features of tag words.
- This is the architecture of the model in keras that predicts the principle direction when the features of an image is given as input
```python
  input_img = Input(shape=(2048,))
  img = Dense(1024, activation = ‘relu’)(input_img)
  img = Dense(512, activation = ‘relu’)(img)
  output = Dense(200)(img)
  model = Model(inputs = [input_img], outputs = [output])
```
- By minimizing the above loss over this model, we can train the network.
- After training, we can input any image feature to get its principle direction.



## SUB-PROBLEM 4 : RANKING AND ASSIGNING TAGS

- Pass an input image to the above trained model to get its Principle direction.
- Then project all the candidate tags (seen + unseen) on this Principle direction, i.e find the inner products of all tags on this Principle direction to get a score of each tag for this image.
- Then sort all the tags by score, to get a ranking of all tags for the particular image
- The top tags will then describe the image well.


