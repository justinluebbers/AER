import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import math

###
# Implementation of the environment models
###

class Approximator:
    def __init__(self, activation = tf.nn.leaky_relu, data_manager = None, hdepth = 10, hwidth = 10,
                model_dir="./AER/models/", scores_dir="./AER/scores/model_training/",
                **kwargs):
        self.hwidth = hwidth
        self.hdepth = hdepth
        self.activation = activation
        self.data_manager = data_manager
        
        self.model_dir = model_dir
        self.scores_dir = scores_dir

    # Build neural network for approximator
    def build_network(self, Z, output_size, dropout = 0., reuse=False, model_name='Approximator'):
        with tf.variable_scope("AER/"+model_name,reuse=reuse):
            tmp = Z
            for _ in range(self.hdepth):
                tmp = tf.layers.dense(tmp,self.hwidth,activation=self.activation)
                if dropout is not 0:
                    tmp = tf.nn.dropout(tmp, rate=dropout)
            out = tf.layers.dense(tmp,output_size)
        return out

    # Train a model
    def train(self, n_steps, loss_function, sample_function, 
            suc_ratio=0.0, learning_rate=.001,
            batch_size=256, dropout=0., model_name='app_MC',
            load_model = None, save_model=True, plot=False, save_losses=False, **kwargs):
        Z = tf.placeholder(tf.float32, [None, self.data_manager.input_size])
        L = tf.placeholder(tf.float32, [None, self.data_manager.output_size])

        if load_model is not None:
            self.depth, self.width = load_model.split('h')[1].split('_')[0].split('x')
        
        P = self.build_network(Z, output_size=self.data_manager.output_size, dropout=dropout, model_name=model_name)

        loss, RMSE = globals()[loss_function](P, L)
        # range_ = tf.reduce_max(L,axis=0)-tf.reduce_min(L,axis=0)
        # MSE = tf.reduce_mean((L-P)**2,axis=0)
        # RMSE = tf.sqrt(MSE)
        # NRMSE = RMSE/range_
        # loss = tf.reduce_sum(NRMSE)
        
        step = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)
        
        sess = tf.Session()
        tf.global_variables_initializer().run(session=sess)
        
        if load_model is not None:
            saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="AER/"+model_name))
            saver.restore(sess, self.model_dir + load_model)

        losses = []
        rmses = []
        done_losses = []

        loss_means = []
        rmse_means = []
        done_means = []
        
        for i in range(n_steps):
            inputs,labels = sample_function(batch_size,suc_ratio)
            
            #_, predictions, loss_,l,r,m,rm,n = sess.run([step, P, loss,L,range_,MSE,RMSE,NRMSE], feed_dict={Z:inputs, L:labels})
            _, predictions, loss_, RMSE_ = sess.run([step, P, loss, RMSE], feed_dict={Z:inputs, L:labels})
            
            losses.append(loss_)
            rmses.append(RMSE_)
            x1 = np.vectorize(round)(predictions[:,0])
            x2 = labels[:,0]
            done_losses.append(abs(x1-x2))
            
            if i%100 == 0:
                loss_mean = sum(losses)/len(losses)
                rmses = np.array(rmses)
                rmse_mean = rmses.mean(axis=0)
                done_mean = np.array(done_losses).mean()

                print('iteration:  {}\tmean_loss: {}'.format(i,loss_mean))
                r = np.random.randint(batch_size)
                print('label:'+str(labels[r])+'\tprediction:'+str(predictions[r]))
                
                loss_means.append(loss_mean)
                rmse_means.append(rmse_mean)
                done_means.append(done_mean)

                losses=[]
                rmses = []
                done_losses = []
    
        if plot:
            plt.plot(loss_means)
            #plt.plot(rmse_means)
            # plt.plot(done_means)
            # plt.legend(range(len(rmse_means[0])))
            plt.legend('error')
            plt.show()

        if save_losses:
            parameters = '_n{}_s{}_h{}x{}_d{}_lr{}_l-{}'.format(str(int(n_steps/1000))+'k' if n_steps >= 1000 else n_steps,
                            suc_ratio,self.hdepth,self.hwidth,dropout,learning_rate,loss_function
                            )
            dir_name = self.scores_dir+model_name+'/'
            if not os.path.exists(dir_name): os.makedirs(dir_name)
            pickle.dump(loss_means, open(dir_name+'loss_means'+parameters+'.p', 'wb'))
            pickle.dump(rmse_means, open(dir_name+'rmse_means'+parameters+'.p', 'wb'))

        # log = open("./AER/Logs/loss_logs_structure_em.csv", 'a')
        # log.write('{},{},{}\n'.format(self.hdepth,self.hwidth,loss_means[-1]))

        if save_model:
            parameters = '_n{}_s{}_h{}x{}_d{}_lr{}_l-{}'.format(str(int(n_steps/1000))+'k' if n_steps >= 1000 else n_steps,
                            suc_ratio,self.hdepth,self.hwidth,dropout,learning_rate,loss_function
                            )
            old_params=''
            if load_model is not None:
                old_params = load_model.split(model_name)[1].split('.ckpt')[0]
            model_filename = model_name+parameters+old_params+'.ckpt'

            v = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="AER/"+model_name)
            v_ = sess.run(v)    
            # print(v_)       

            saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="AER/"+model_name))
            path = saver.save(sess, self.model_dir + model_filename)


        tf.reset_default_graph()
        sess.close()

        return loss_means

    # Load the perfect environment model instead of a trained one
    def load_perfect_session(self, model_name, input_size):
        model_type, env = model_name.split('_')
        if env == 'MountainCar':
            return self.perfect_MountainCar(model_type, model_name, input_size)
        if env == 'CartPole':
            return self.perfect_CartPole(model_type, model_name, input_size)
        else:
            return None, None

    # Perfect MountainCar dynamics
    def perfect_MountainCar(self, model_type, model_name, input_size):
        if model_type == 'RM':
            with tf.variable_scope("AER/"+model_name,reuse=False):
                RM_Z = tf.placeholder(tf.float32, shape=[None, input_size])
                RM = tf.add(tf.matmul(RM_Z, np.zeros(shape=[input_size,1], dtype=np.float32)), -1.)
            return RM, RM_Z
        elif model_type == 'TM':
            with tf.variable_scope("AER/"+model_name,reuse=False):
                TM_Z = tf.placeholder(tf.float32, shape=[None, input_size])
                cond = tf.math.logical_and(tf.math.greater_equal(TM_Z[:,0], tf.constant(0.5)), tf.math.greater_equal(TM_Z[:,1],tf.constant(0.)))
                TM = tf.where(cond, tf.ones_like(TM_Z[:,0]), tf.zeros_like(TM_Z[:,0]))
            return TM, TM_Z
        elif model_type == 'EM':
            with tf.variable_scope("AER/"+model_name,reuse=False):
                EM_Z = tf.placeholder(tf.float32, shape=[None, input_size])
                pos = EM_Z[:,0]
                vel = EM_Z[:,1]
                act = EM_Z[:,2]
                
                tmp1 = tf.math.subtract(act, tf.constant(1.))
                tmp1 = tf.math.multiply(tmp1, tf.constant(0.001))
                tmp2 = tf.math.cos(tf.math.multiply(tf.constant(3.), pos))
                tmp2 = tf.math.multiply(tmp2, tf.constant(-0.0025))
                tmp3 = tf.math.add(tmp1, tmp2)
                new_vel = tf.math.add(vel, tmp3)
                new_vel = tf.clip_by_value(new_vel, tf.constant(-0.07), tf.constant(0.07))

                new_pos = tf.add(pos, new_vel)
                new_pos = tf.clip_by_value(new_pos, tf.constant(-1.2), tf.constant(0.6))
                
                cond = tf.logical_and(tf.math.equal(new_pos,tf.constant(-1.2)),tf.math.less(new_vel,tf.constant(0.)))

                new_vel = tf.where(cond, tf.zeros_like(new_vel), new_vel)
                
                EM = tf.stack([new_pos, new_vel], 1)
            return EM, EM_Z
        else:
            return None, None

    # Perfect CartPole dynamics
    def perfect_CartPole(self, model_type, model_name, input_size):
        if model_type == 'RM':
            with tf.variable_scope("AER/"+model_name,reuse=False):
                RM_Z = tf.placeholder(tf.float32, shape=[None, input_size])
                RM = tf.add(tf.matmul(RM_Z, np.zeros(shape=[input_size,1], dtype=np.float32)), 1.)
            return RM, RM_Z
        elif model_type == 'TM':
            with tf.variable_scope("AER/"+model_name,reuse=False):
                TM_Z = tf.placeholder(tf.float32, shape=[None, input_size])
                x = TM_Z[:,0]
                theta = TM_Z[:,2]

                cond1 = tf.logical_or(tf.math.less(x,tf.constant(-2.4)),tf.math.greater(x,tf.constant(2.4)))
                angle = 12 * 2 * math.pi / 360
                cond2 = tf.logical_or(tf.math.less(theta,tf.constant(-angle)),tf.math.greater(theta,tf.constant(angle)))
                cond = tf.logical_or(cond1, cond2) 
                
                TM = tf.where(cond, tf.ones_like(TM_Z[:,0]), tf.zeros_like(TM_Z[:,0]))

            return TM, TM_Z
        elif model_type == 'EM':
            TM_Z = tf.placeholder(tf.float32, shape=[None, input_size])
            x = TM_Z[:,0]
            x_dot = TM_Z[:,1]
            theta = TM_Z[:,2]
            theta_dot = TM_Z[:,3]
            action = TM_Z[:,4]

            force = tf.where(tf.equal(action, tf.ones_like(action)), 10.0*tf.ones_like(action), -10.0*tf.ones_like(action))
            costheta = tf.math.cos(theta)
            sintheta = tf.math.sin(theta)

            temp = (force + 0.05 * theta_dot**2 * sintheta) / 1.1
            thetaacc = (9.8 * sintheta - costheta * temp) / (0.5 * (4.0/3.0 - 0.1 * costheta ** 2 / 1.1))
            xacc = temp - 0.05 * thetaacc * costheta / 1.1

            x = x + 0.02 * x_dot
            x_dot = x_dot + 0.02 * xacc
            theta = theta + 0.02 * theta_dot
            theta_dot = theta_dot + 0.02 * thetaacc

            TM = tf.stack([x,x_dot,theta,theta_dot], 1) 
            return TM, TM_Z
        else:
            return None, None

    # Perfect Acrobot dynamics - unfinished
    def perfect_Acrobot(self, model_type, model_name, input_size):
        if model_type == 'RM':
            return None, None
        elif model_type == 'TM':
            return None, None
        elif model_type == 'EM':
            TM_Z = tf.placeholder(tf.float32, shape=[None, input_size])
            
        else:
            return None, None

    # Load Approximator from file
    def load_session(self, sess, filename, input_size, output_size):
        Z = tf.placeholder(tf.float32,[None,input_size])

        model_name=filename.split('_')[0:2]
        model_type = model_name[0]
        model_name = model_name[0]+'_'+model_name[1]

        perfect = filename.split('.')[-1] == 'perfect'
       
        if not perfect:
            approximator = self.build_network(Z, output_size=output_size, model_name=model_name)
            saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="AER/"+model_name))
            saver.restore(sess, self.model_dir + filename)
            if model_name.startswith('TM'):
                approximator = tf.round(tf.nn.sigmoid(approximator))
            return approximator, Z
        
        else:
            return self.load_perfect_session(model_name, input_size)

    # Run the model to predict values
    def predict(self, observations):
        return self.sess.run(self.approximator, feed_dict={self.Z:observations})


# Possible loss functions for training

def termination_loss(P,L):
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=L, logits=P)
    loss = tf.reduce_mean(cross_entropy)
    return loss, cross_entropy
    
def nrmse_range(P, L):
    range_ = tf.reduce_max(L,axis=0)-tf.reduce_min(L,axis=0)
    range_ = tf.where(tf.equal(range_,0), tf.ones_like(range_), range_)
    MSE = tf.reduce_mean((L-P)**2,axis=0)
    RMSE = tf.sqrt(MSE)
    NRMSE = RMSE/range_
    return tf.reduce_sum(NRMSE), NRMSE

def nrmse_mean(P, L):
    mean_ = tf.abs(tf.reduce_mean(L,axis=0))
    MSE = tf.reduce_mean((L-P)**2,axis=0)
    RMSE = tf.sqrt(MSE)
    NRMSE = RMSE/mean_
    return tf.reduce_sum(NRMSE), RMSE

def rmse(P,L):
    MSE = tf.reduce_mean((L-P)**2,axis=0)
    RMSE = tf.sqrt(MSE)
    return tf.reduce_sum(RMSE), RMSE

def mae(P,L):
    MAE = tf.reduce_mean(tf.abs(L-P),axis=0)
    return tf.reduce_sum(MAE), MAE

def nmae_range(P,L):
    range_ = tf.reduce_max(L,axis=0)-tf.reduce_min(L,axis=0)
    range_ = tf.where(tf.equal(range_,0), tf.ones_like(range_), range_)
    MAE = tf.reduce_mean(tf.abs(L-P),axis=0)
    NMAE = MAE/range_
    return tf.reduce_sum(NMAE), NMAE

        