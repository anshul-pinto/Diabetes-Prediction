
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import csv


# In[2]:


X,Y=[],[]
#reading the file
def getdata(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        
        for row in csvFileReader:
            X.append([[float(row[0])],[float(row[1])],[float(row[2])],[float(row[3])],[float(row[4])],[float(row[5])],[float(row[6])],[float(row[7])]])
            Y.append(float(row[8]))
    
    return X,Y        


# In[3]:


X,Y = getdata('diabetes.csv')
#have the X and Y ready, time to make training and testing set ready 


# In[4]:


X=np.reshape(X,(768,8))
Y=np.reshape(Y,(768,1))

#normalizing inputs

X_max = (np.max(X, axis=0))

for i in range(768):
    for j in range(8):
        X[i][j]= X[i][j]/X_max[j]


# In[5]:


X_train, X_test, Y_train, Y_test = X[:700],X[700:],Y[:700],Y[700:]
X_train, Y_train = X_train.T,Y_train.T
X_test, Y_test = X_test.T,Y_test.T


# In[6]:


print(np.shape(X_train),np.shape(Y_train))
print(np.shape(X_test),np.shape(Y_test))


# In[13]:



#make all parameters and hyper parameter variables
n_1 = 8
n_2 = 5
n_3 = 3
n_out = 1

input_to_nn_X = tf.placeholder(tf.float32, shape=(8,None)) 
input_to_nn_Y = tf.placeholder(tf.float32, shape=(1,None))


# In[14]:


#making the neural network
#4-Layer neural net
def neural_network(x):
    #initializing weight and bias matrices
    hidden_layer_1 = {'W1': tf.Variable(tf.random_normal([n_1,8]))*0.01,
                      'b1': tf.Variable(tf.zeros([n_1,1]))}
    
    hidden_layer_2 = {'W2': tf.Variable(tf.random_normal([n_2,n_1]))*0.01,
                      'b2': tf.Variable(tf.zeros([n_2,1]))}
    
    hidden_layer_3 = {'W3': tf.Variable(tf.random_normal([n_3,n_2]))*0.01,
                      'b3': tf.Variable(tf.zeros([n_3,1]))}
    
    output_layer = {'W4': tf.Variable(tf.random_normal([n_out,n_3]))*0.01,
                    'b4': tf.Variable(tf.zeros([n_out,1]))}
    
    #computing neurons' matrix multiplications
    #passing through activation functions relu-->relu-->relu-->sigmoid
    Z1 = tf.add(tf.matmul(hidden_layer_1['W1'],input_to_nn_X),hidden_layer_1['b1'])
    A1 = tf.nn.relu(Z1)
   
    Z2 = tf.matmul(hidden_layer_2['W2'],A1)+ hidden_layer_2['b2']
    A2 = tf.nn.relu(Z2)
    
    Z3 = tf.matmul(hidden_layer_3['W3'],A2)+ hidden_layer_3['b3']
    A3 = tf.nn.relu(Z3)
    
    Z4 = tf.matmul(output_layer['W4'],A3) + output_layer['b4']
    A4 = tf.nn.sigmoid(Z4)
    
    return A4
    


# In[39]:


def train_neural_network(x):
    prediction = neural_network(x)
    cost = tf.losses.mean_squared_error(input_to_nn_Y,prediction)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(cost)
    
    epochs = 75000
    hm_correct=0
    hm_training_correct=0
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        for epoch in range(epochs):
            epoch_loss = 0
            _,c = sess.run([optimizer,cost], feed_dict={input_to_nn_X:X_train,
                                                  input_to_nn_Y:Y_train})
            epoch_loss+=c
            if epoch%5000==0:
                print('Epoch', epoch+5000,'completed out of ',epochs,' loss:', epoch_loss)
            
        correct = sess.run(prediction, feed_dict={input_to_nn_X:X_test,
                                                  input_to_nn_Y:Y_test})
        
        for i in range(68):
            if correct[0][i]<0.5:
                correct[0][i]=0
            else:
                correct[0][i]=1
                
            if correct[0][i]==Y_test[0][i]:
                hm_correct+=1
        
        training_correct = sess.run(prediction,feed_dict={input_to_nn_X:X_train,
                                                  input_to_nn_Y:Y_train})
        for i in range(700):
            if training_correct[0][i]<0.5:
                training_correct[0][i]=0
            else:
                training_correct[0][i]=1
                
            if training_correct[0][i]==Y_train[0][i]:
                hm_training_correct+=1
        print('Training accuracy: ' + str((hm_training_correct/700)*100)+ '%')        
        print('Testing accuracy : '+str((hm_correct/68)*100) + '%')


# In[40]:


train_neural_network(input_to_nn_X)

