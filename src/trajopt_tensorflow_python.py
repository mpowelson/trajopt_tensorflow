from __future__ import print_function
import tensorflow as tf

## Define function to setup the problem
def setupProblem(dof, steps):
  with tf.variable_scope("TrajOpt", reuse=tf.AUTO_REUSE):
    v = tf.get_variable("joint_vars", [dof, steps])
  return v

## Define our functions to generate the variables
def getJointVars():
  with tf.variable_scope("TrajOpt", reuse=tf.AUTO_REUSE):
    v = tf.get_variable("joint_vars")
  return v

def jointVelocityCost():
    joint_vals = getJointVars()
    diff0 = tf.slice(joint_vals, [0,0], tf.shape(joint_vals) - [1,0])
    diff1 = tf.slice(joint_vals, [1,0], tf.shape(joint_vals) - [1,0])
    diff = diff1 - diff0
    cost = tf.reduce_sum(tf.square(diff))
    return cost

def fixStartCost():
    joint_vals2 = getJointVars()
    fixed_value2 = tf.constant([0.,0.,0.,0.,0.,0.])
    first_row2 = tf.slice(joint_vals2, [0,0], [1,6])
    cost = tf.reduce_sum(tf.square((first_row2-fixed_value2)))
    return cost

def fixEndCost():
    joint_vals3 = getJointVars()
    fixed_value3 = tf.constant([1.,1.,1.,1.,1.,1.])
    last_row3 = tf.slice(joint_vals3, [9,0], [1,6])
    cost = tf.reduce_sum(tf.square((last_row3-fixed_value3)))
    return cost

class TensorflowProblem:

    cost = None
    def __init__(self, cost_in):
        print("Initializing the Python TensorflowProblem class")
        self.cost = cost_in


    def addCost(self, to_add):
        self.cost = self.cost + to_add


    def solveProblem(self):
        joint_vals = getJointVars()
        cost_grad = tf.gradients(self.cost, [joint_vals])
        cost_grad2 = tf.gradients(cost_grad, [joint_vals])

        # Now we optimize
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(self.cost)
        # https://www.tensorflow.org/api_docs/python/tf/train/Optimizer

        model = tf.global_variables_initializer()

        print("Optimizing...")
        with tf.Session() as session:
            session.run(model)
            for i in range(10000):
                session.run(train_op)

            w_value = session.run(getJointVars())
            print("Joint Values: ")
            print(w_value)
            print("Cost:")
            print(session.run(self.cost))
            print("Cost Gradient:")
            print(session.run(cost_grad))
            print("Cost Gradient2:")
            print(session.run(cost_grad2))

            print("Returning Python type: ")
            print(type(w_value))
            return w_value




### Setup the problem
#print("Setting up problem...")
#print(setupProblem(10,6))

### Now we write out a cost (joint velocity squared error)
#print("Defining the costs...")
#cost_val_1 = jointVelocityCost()
#print(cost_val_1)

## Now we write out another cost (Fix the start position)
#cost_val_2 = fixStartCost()
#print(cost_val_2)

## Now we write out another cost (Fix the end position)
#cost_val_3 = fixEndCost()
#print(cost_val_3)

## Now we join all of the costs
#print("Joining the costs...")
#cost = cost_val_1 + 10*cost_val_2 + 10*cost_val_3
#print(cost)

## Calculate the gradients just for fun
#joint_vals = getJointVars()
#cost_grad = tf.gradients(cost, [joint_vals])
#cost_grad2 = tf.gradients(cost_grad, [joint_vals])

## Now we optimize
#train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
## https://www.tensorflow.org/api_docs/python/tf/train/Optimizer

#model = tf.global_variables_initializer()

#print("Optimizing...")
#with tf.Session() as session:
#    session.run(model)
#    for i in range(10000):
#        session.run(train_op)

#    w_value = session.run(getJointVars())
#    print("Joint Values: ")
#    print(w_value)
#    print("Cost:")
#    print(session.run(cost))
#    print("Cost Gradient:")
#    print(session.run(cost_grad))
#    print("Cost Gradient2:")
#    print(session.run(cost_grad2))
