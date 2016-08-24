# import converted model's class
import os
import numpy as np
import random
import tensorflow as tf
import h5py
from mynet import AlexNet as MyNet
from retrieval import retrieval

# define Image shape
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_CHANNEL = 3

# define Image Fully-connected layer size
FC6_SIZE = 4096
FC7_SIZE = 4096

# batch_size
batch_size = 16

# tag_size
TAG_SIZE = 21

# binary code length
binary_length = 8

# Text network input
text_input = 21
# Text network fc6 units number
n_fc6 = 256
# Text network fc7 units number
n_fc7 = 256

# hdf5 file path
Data_Path = r"../hdf5/Train/"
Val_Path = r"../hdf5/Validation/hdf5.h5"

summaries_dir = r"../hdf5"
model_dir = r"../hdf5/model"
param_dir = r"../hdf5/parameter.txt"

Text_param = {
 "fc6_weights": tf.Variable(tf.truncated_normal([text_input,n_fc6],stddev=0.001),name='fc6_weights'),
 "fc6_biases" : tf.Variable(np.zeros([n_fc6]).astype(np.float32),name="fc6_biases"),
 "fc7_weights": tf.Variable(tf.truncated_normal([n_fc6,n_fc7],stddev=0.001),name='fc7_weights'),
 "fc7_biases" : tf.Variable(np.zeros([n_fc7]).astype(np.float32),name="fc7_biases"),
 "fc8_weights": tf.Variable(tf.truncated_normal([n_fc7,binary_length],stddev=0.001),name='fc8_weights'),
 "fc8_biases" : tf.Variable(np.zeros([binary_length]).astype(np.float32),name="fc8_biases")
}

# use 21 categories to train our fine tune model
category = ["building","clouds","flowers","grass","lake","person","plants","sky","water","window","beach","birds","boats", "military","mountain","tree","reflection","road","rocks","sunset","vehicle"]
#category = ["building"]
# max_iteration
MAX_ITERATION = 100

# read hdf5 file 
def readhdf5(hdf5file):
    with h5py.File(hdf5file,'r') as hf:
        print "Read from",hdf5file
#       print "List of items in the base directory",hf.items()
        tag = np.array(hf.get('Tag'))
        img = np.array(hf.get('data'))
        hf.close()
    return img , tag

def gen_data(img,tag):
    while True:
        indices = range(len(img))
        random.shuffle(indices)
        for i in indices:
            image = np.reshape(img[i], (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))
            #image = tf.cast(image, tf.float32)
            t = tag[i]
            #tag = tf.cast(tag, tf.float32)
            yield image, t

def gen_data_batch(img,tag):
    data_gen = gen_data(img,tag)
    while True:
        image_batch = []
        tag_batch = []
        for _ in range(batch_size):
            image, tag = next(data_gen)
            image_batch.append(image)
            tag_batch.append(tag)
#        print np.array(image_batch).shape
#        print np.array(tag_batch).shape
        yield np.array(image_batch), np.array(tag_batch)

def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.scalar_summary('sttdev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)


# Add new train layer to Image model
def add_final_training_op(bottleneck_tensor):
    """ Add three full-connected layer to Image model
    
    We need to retrain the top layer to fit our task, so this function
    adds the right operations to the graph, along with some variables to hold the
    weights, and then sets up all the gradients for the backward pass.
    
    Args:
        bottleneck_tensor: The output of main CNN graph
        
    Returns:
        The tensor for the training results.
    """
    # Organizaing the following op as "final_training_op" so they are easy to see
    # in the tensorBoard
    layer_name = "Image_final_training_ops"
    with tf.name_scope(layer_name):
        with tf.name_scope("weights"):
            fc8_weights = tf.Variable(tf.truncated_normal([FC7_SIZE,binary_length],stddev=0.001),name='final_weights')
            variable_summaries(fc8_weights,layer_name+"/weights")
        with tf.name_scope("biases"):
            fc8_biases = tf.Variable(np.zeros([binary_length]).astype(np.float32), name='final_biases')
            variable_summaries(fc8_biases,layer_name+"/biases")
        with tf.name_scope("Wx_Plus_b"):
            logits = tf.matmul(bottleneck_tensor,fc8_weights) + fc8_biases
            logits = tf.nn.tanh(logits)
            tf.histogram_summary(layer_name + '/Tanh_activations', logits)
     
    return logits

# Create MLP model
def multilayer_perceptron(x):
    # Text network fc6 layers
    layer_name = "Text_FC6"
    with tf.name_scope(layer_name):
	with tf.name_scope("weights"):
            fc6_weights = Text_param["fc6_weights"]
            variable_summaries(fc6_weights,layer_name+"/weights")
        with tf.name_scope("baises"):
            fc6_biases = Text_param["fc6_biases"]
            variable_summaries(fc6_biases,layer_name+"/biases")
	with tf.name_scope("Wx_Plus_b"):
            fc6 = tf.add(tf.matmul(x,fc6_weights),fc6_biases)
            fc6 = tf.nn.relu(fc6)
            tf.histogram_summary(layer_name+"/ReLu_activations",fc6)
    
    # Text network fc7 layers
    layer_name = "Text_FC7"
    with tf.name_scope(layer_name):
        with tf.name_scope("weights"):
            fc7_weights = Text_param["fc7_weights"]
            variable_summaries(fc7_weights,layer_name+"/weights")
        with tf.name_scope("biases"):
            fc7_biases = Text_param["fc7_biases"]
            variable_summaries(fc7_biases,layer_name+"/biases")
        with tf.name_scope("Wx_Plus_b"):
            fc7 = tf.add(tf.matmul(fc6,fc7_weights),fc7_biases)
            fc7 = tf.nn.relu(fc7)
            tf.histogram_summary(layer_name+"/ReLu_activations",fc7)
    
    # Text network fc8 layers with tanh activative function
    layer_name = "Text_FC8"
    with tf.name_scope(layer_name):
        with tf.name_scope("weights"):
            fc8_weights = Text_param["fc8_weights"]
            variable_summaries(fc8_weights,layer_name+"/weights")
        with tf.name_scope("biases"):
            fc8_biases = Text_param["fc8_biases"]
            variable_summaries(fc8_biases,layer_name+"biases")
        with tf.name_scope("Wx_Plus_b"):
            out_layer = tf.add(tf.matmul(fc7,fc8_weights),fc8_biases)
            out_layer = tf.nn.tanh(out_layer)
            tf.histogram_summary(layer_name+"/Tanh_activations",out_layer)
    return out_layer

def cosin(x,y):
    """Define cosin function
    
        args:
            x: input vector,shape=[batch_size, binary_length]
            y: input vector,shape=[batch_size, binary_length]
        
        return:
            cosin result. shape=[batch_size,]
    """
    inner = tf.reduce_sum(tf.mul(x,y),1)
    x_length = tf.sqrt(tf.reduce_sum(tf.square(x),1))
    y_length = tf.sqrt(tf.reduce_sum(tf.square(y),1))
    x_y_length = tf.mul(x_length,y_length)
    cosin = tf.div(inner, x_y_length)
    return cosin

def _loss(image_u1,tag_v1,image_u2,tag_v2,similarity,alpha,gamma):
    """Add Loss to all trainable variables
    
        args:
            image_u: Image Network input
            tag_v: Tag Network input
            alpha: hyper-parameter
            gamma: hyper-paramter
            
        output:
            loss: Return cross-modal loss 
    """
    # Calculate cos(u1,v2)
    cos_u_v = cosin(image_u1,tag_v2)
    
    # Calculate cos(v1,u2)
    cos_v_u = cosin(tag_v1,image_u2)
    
    # Cross-Modal Correlation
    cross_u_v = tf.square(tf.sub(similarity,cos_u_v))
    cross_v_u = tf.square(tf.sub(similarity,cos_v_u))
    Cxy = tf.add(cross_u_v,cross_v_u)
    
    # Calculate cos(u1,u2)
    cos_u_u = cosin(image_u1,image_u2)
    cos_v_v = cosin(tag_v1,tag_v2)
    
    # Within-Modal Correlation
    Cxx = tf.square(tf.sub(similarity,cos_u_u))
    Cyy = tf.square(tf.sub(similarity,cos_v_v))
    
    one = tf.constant(1,dtype=tf.float32,shape=[batch_size,binary_length])
    neg_one = tf.constant(-1,dtype=tf.float32,shape=[1])
    
    # Cosine Quantization Loss
    Qx = tf.mul(neg_one,tf.add(cosin(tf.abs(image_u1),one),cosin(tf.abs(image_u2),one)))
    Qy = tf.mul(neg_one,tf.add(cosin(tf.abs(tag_v1),one),cosin(tf.abs(tag_v2),one)))
    
    within_loss = tf.mul(alpha,tf.add(Cxx,Cyy))
    quan_loss = tf.mul(gamma,tf.add(Qx,Qy))
    loss = tf.reduce_mean(tf.add(Cxy,tf.add(within_loss,quan_loss)),0)
    
    return loss

def sgn(x):
    row , col =x.shape
    for i in range(row):
        for j in range(col):
            if x[i,j]>0:
                x[i,j] = 1
            else:
                x[i,j] = -1
    return x
    
    
def main(_):
    # Graph input
    images = tf.placeholder(tf.float32,[batch_size,IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNEL])
    tags = tf.placeholder(tf.float32,[batch_size,TAG_SIZE])
    similarity = tf.placeholder(tf.float32,[batch_size])
    alpha = tf.placeholder(tf.float32,[1])
    gamma = tf.placeholder(tf.float32,[1])
    learning_rate = tf.placeholder(tf.float32)
    #momentum = tf.placeholder(tf.float32,[1])
    
    # Contruct Text network
    Tag_v =  multilayer_perceptron(tags)

    # Construct Image network
    net = MyNet({'data':images})
    
    image_fc7 = net.layers['fc7']
    
    Image_u = add_final_training_op(image_fc7)
#    
#    Tag_v1,Tag_v2 = tf.split(0,2,Tag_v)
#    Image_u1,Image_u2 = tf.split(0,2,Image_u)
#    np_images1,np_tags1 = next(data_gen1)

#    loss = _loss(Image_u1,Tag_v1,Image_u2,Tag_v2,similarity,alpha,gamma)
#    
#    opt = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
#        
#    # Add ops to save and restore all the variable.
#    saver = tf.train.Saver()
#    
#    
    with tf.Session() as sess:
        # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
#        merged = tf.merge_all_summaries()
#        train_writer = tf.train.SummaryWriter(summaries_dir + '/summary_train',sess.graph)
        # Load the data
        sess.run(tf.initialize_all_variables())
        print "load net"
        net.load('mynet.npy',sess)

	image_hd,tag_hd = readhdf5(Data_Path+cate1+"/hdf5.h5")
	data_gen = gen_data_batch(image_hd,tag_hd) 

	for i in xrange(MAX_ITERATION):
            np_images,np_tags = next(data_gen)
	    feed = {images: np_images,tags: np_tags}
	    sess.run([Tag_v,Image_u], feed_dict=feed)

        
        # reset fc6 and fc7 layers' weights and biases
#        ignore_missing = False
#        with tf.variable_scope("fc6",reuse=True):
#            try:
#                weights = tf.get_variable("weights")
#                weight_data = np.random.randn(9216,4096).astype(np.float32)
#                sess.run(weights.assign(weight_data))
#                biases = tf.get_variable("biases")
#                biase_data = np.zeros(4096).astype(np.float32)
#                sess.run(biases.assign(biase_data))
#            except ValueError:
#                if not ignore_missing:
#                    raise
#         
#        with tf.variable_scope("fc7",reuse=True):
#            try:
#                weights = tf.get_variable("weights")
#                weight_data = np.random.randn(4096,4096).astype(np.float32)
#                sess.run(weights.assign(weight_data))
#                biases = tf.get_variable("biases")
#                biase_data = np.zeros(4096).astype(np.float32)
#                sess.run(biases.assign(biase_data))
#            except  ValueError:
#                if not ignore_missing:
#                    raise
#                    
#        max_count = 100 
#        best_map_image = 0
#        best_map_text = 0
#        
#        for count in xrange(max_count):
#            lr = 10**random.uniform(-5,0)
#            mom = 0.9
#            al  = random.uniform(-10,10)
#            ga = random.uniform(-10,10)
#            
#            total_step = 0 
#        
#            # We use two iteration to imitation cross-modal correlationship
#            for cate1 in category:
#                for cate2 in category:
#                    image_hd1,tag_hd1 = readhdf5(Data_Path+cate1+"/hdf5.h5")
#                    image_hd2,tag_hd2 = readhdf5(Data_Path+cate2+"/hdf5.h5")
#
#
#                    # read batch_data 
#                    data_gen1 = gen_data_batch(image_hd1,tag_hd1) 
#                    data_gen2 = gen_data_batch(image_hd2,tag_hd2)
#
#                    if cate1==cate2:
#                        sim = np.ones(batch_size).astype(np.float32)
#                    else:
#                        sim = - np.ones(batch_size).astype(np.float32)
#
#                    for i in range(MAX_ITERATION):
#                        np_images1,np_tags1 = next(data_gen1)
#                        np_images2,np_tags2 = next(data_gen2)
#
#                        np_images = np.vstack((np_images1,np_images2))
#                        np_tags = np.vstack((np_tags1,np_tags2))
#
#                        #l = np.array([lr]).astype(np.float32)
#                        a = np.array([al]).astype(np.float32)
#                        g = np.array([ga]).astype(np.float32)
#
#                        feed = {images: np_images,tags: np_tags,similarity: sim,alpha: a,gamma: g, learning_rate: lr}
#
#                        summary_str,np_loss, _ = sess.run([merged,loss, opt], feed_dict=feed)
#                        
#                        total_step += 1
#                        
#                        if total_step%100 == 0:
#			   # print "weights"
#			   # print sess.run(Text_param["fc6_weights"])
#                            print "Iteration: %d; loss: %f" % (total_step,np_loss)
#                            train_writer.add_summary(summary_str,total_step)
#                            
#            # Validation our model result
#            image_val, tag_val = readhdf5(Val_Path)
#            length = image_val.shape[0]
#            times = length/(2*batch_size)
#            image_code = np.zeros([times*2*batch_size,binary_length]).astype(np.float32)
#            text_code = np.zeros([times*2*batch_size,binary_length]).astype(np.float32)
#            for i in range(times):
#                feed = {images:image_val[i*2*batch_size : (i+1)*2*batch_size], tags:tag_val[i*2*batch_size : (i+1)*2*batch_size]}
#                image_code[i*2*batch_size : (i+1)*2*batch_size], text_code[i*2*batch_size : (i+1)*2*batch_size] = sess.run([Image_u,Tag_v], feed_dict=feed)
#
#	    #for i in xrange(image_code.shape[0]):
#             #   print "example:",i
#	     #	print image_val[i]
#             #   print image_code[i]
#             #   print text_code[i]
#             #   print tag_val[i]
#                
#            image_code = sgn(image_code)
#            text_code = sgn(text_code)
#            
#            #for i in xrange(image_code.shape[0]):
#		#print "example:",i
#		#print image_code[i]
#		#print text_code[i]
#	    	#print tag_val[i]
#		    
#            _,_,img_info = retrieval(image_code,tag_val[0 : times*2*batch_size],text_code,tag_val[0 : times*2*batch_size],True)
#            _,_,txt_info = retrieval(text_code,tag_val[0 : times*2*batch_size],image_code,tag_val[0 : times*2*batch_size],True)
#            img_mAP = img_info["mAP"] 
#            if img_mAP > best_map_image:
#                best_map_image = img_mAP
#                print 'learning_rate: %f; alpha: %f; gamma: %f; image_retrieval_text_map: %f' % (lr,al,ga,img_mAP)
#                # Save the variable to disk
#                save_path = saver.save(sess, model_dir+"/img_retr_txt_model.ckpt")
#                print "Model saved in file: %s" % save_path
#                
#	    txt_mAP = txt_info["mAP"]
#            if txt_mAP > best_map_text:
#                best_map_text = txt_mAP
#                print 'learning_rate: %f; alpha: %f; gamma: %f; text_retrieval_image_map: %f' % (lr,al,ga,txt_mAP) 
#                # Save the variable to disk
#                save_path = saver.save(sess, model_dir+"/txt_retr_img_model.ckpt")
#                print "Model saved in file: %s" % save_path
#	    f = file(param_dir,'a')
#	    f.write('learning_rate: %f; alpha: %f; gamma: %f; image_retrieval_text_map: %f; text_retrieval_image_map:%f \n' % (lr,al,ga,img_mAP,txt_mAP))
#	    f.close()
            
                        
if __name__ == '__main__':
    tf.app.run()
