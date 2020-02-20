from __future__ import print_function
import tensorflow as tf
import os
import timeit


#%% Define function to setup the problem
@tf.function
def init_vars(steps, dof):
    #v = tf.Variable([steps, dof])
    v = tf.zeros([steps, dof])
    return v

@tf.function
def joint_velocity_cost(joint_vals):
    diff0 = tf.slice(joint_vals, [0,0], tf.shape(joint_vals) - [1,0])
    diff1 = tf.slice(joint_vals, [1,0], tf.shape(joint_vals) - [1,0])
    diff = diff1 - diff0
#    cost = tf.reduce_sum(tf.square(diff))
    cost = tf.square(diff)
    return cost

@tf.function
def fix_start_cost(joint_vals):
    fixed_value2 = tf.constant([0.,0.,0.,0.,0.,0.])
    first_row2 = tf.slice(joint_vals, [0,0], [1,6])
    tf.print(first_row2-fixed_value2)
#    cost = tf.reduce_sum(tf.square((first_row2-fixed_value2)))
    cost = tf.square((first_row2-fixed_value2))
    return cost

@tf.function
def fix_end_cost(joint_vals):
    fixed_value3 = tf.constant([1.,1.,1.,1.,1.,1.])
    last_row3 = tf.slice(joint_vals, [9,0], [1,6])
    tf.print(last_row3-fixed_value3)
#    cost = tf.reduce_sum(tf.square((last_row3-fixed_value3)))
    cost = tf.square((last_row3-fixed_value3))
    return cost

#%%
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


## Setup the problem
print("Setting up problem...")
#vars = init_vars(10, 6)
vars = tf.Variable(tf.zeros([10,6]))
print(vars)

with tf.GradientTape() as tape:
    # Now we write out a cost (joint velocity squared error)
    print("Defining the costs...")
    cost_val_1 = joint_velocity_cost(vars)
    print(cost_val_1)

    # Now we write out another cost (Fix the start position)
    cost_val_2 = fix_start_cost(vars)
    print(cost_val_2)

    # Now we write out another cost (Fix the end position)
    cost_val_3 = fix_end_cost(vars)
    print(cost_val_3)

    # Now we join all of the costs
    print("Joining the costs...")
    cost = cost_val_1 + 10*cost_val_2 + 10*cost_val_3
    print(cost)

print("Calculating gradient")
grad = tape.gradient(cost, vars)
print(grad)




print("Tensorflow version is {0}".format(tf.__version__))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print("Benchmarking speed...")
def cost_func():
    return fix_start_cost(vars) + joint_velocity_cost(vars) + fix_end_cost(vars)
print("Function:", timeit.timeit(lambda: cost_func(), number=100))
tf.print(cost)

# Now we optimize
opt = tf.keras.optimizers.SGD(learning_rate=0.01)
for i in range(100):
    a = opt.minimize(cost_func, vars)

print("Joint Values: ")
tf.print(vars, summarize=50)
#print("Cost:")
#print(cost_func)
#print("Cost Gradient:")
#print(grad)






#cost_grad = tf.gradients(cost, [vars])
#print(cost_grad)



#with tf.Session() as session:
#    print(vars.eval())
#    print(joint_velocity_cost(vars).eval())
    

# print("Optimizing...")
# with tf.Session() as session:
#     session.run(model)
#     for i in range(10000):
#         session.run(train_op)
#
#     w_value = session.run(getJointVars())
#     print("Joint Values: ")
#     print(w_value)
#     print("Cost:")
#     print(session.run(cost))
#     print("Cost Gradient:")
#     print(session.run(cost_grad))
#     print("Cost Gradient2:")
#     print(session.run(cost_grad2))
