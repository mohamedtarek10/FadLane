import tensorflow as tf

def weight_scalling_factor(clients_trn_data):
    '''
    Return the Factor of scalling for each client based on the number of data samples trained on.
    Inputs : clients_trn_data --> Train data for clients
    Outputs: Scaling Factor for the clients weights
    '''

    #get the bs
    bs = 35
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data).numpy()*bs
    return local_count/3626


def scale_model_weights(weight, scalar):
    '''
    function for scaling a models weights
    Inputs: weights --> weights of the client model
            scalar --> Scalling factor of the client factor

    Outputs: weight_final --> updated scaled weights
    '''
    
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final



def sum_scaled_weights(scaled_weight_list):
    '''
    Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights
    Inputs: scaled_weight_list --> list of Factorized clients weights
    Outputs: avg_grad --> Summation of factorized weights to represent the global model
    '''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad
