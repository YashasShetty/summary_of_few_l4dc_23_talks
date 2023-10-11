# Summary_of_few_l4dc_23_talks

Summary by : Rohit Reddy and Yashas Shetty

# Introduction

The Learning for Dynamics & Control Conference is an annual conference which focusses on new finfdngs in the field of control and dynamical systems. 
It is interdisciplinary, bringing together researchers from control, robotics, machine learning, and optimization
One of the goals of L4DC is to build strong ties between these disciplines and enable active collaboration 
The 5th Annual Learning for Dynamics & Control Conference (L4DC) was held from 14 June -16 June 2023 at the University of Pennsylvania.
We are going to summarize the following topics from the L4DC 2023 conference 

1. Model Predictive Control via On-Policy Imitation Learning 
2. Policy Learning for Active Target Tracking over Continuous Trajectories 
3. Policy Evaluation in Distributional LQR 
4. Agile Catching with Whole-Body MPC and Blackbox Policy Learning 

We decided to summarize a few talks related to policy learning, policy evualtion. As we can see the first talk brings out a novice method where we overcome the issue of complicated computations in MPC by using imititation learning . A policy learning method for active target tracking in a continuous domain has been explained. There is another novice method to calculate a dsitrbutional LQR with  the help of policy learning. We then have a comparison between a Sequential Quadratic Programming solver and Blacbox policy learning method in robot arm catching application. The common theme of these topics is policy learning. We explored different types of policy learning presented in the conference.

Source :<https://l4dc.seas.upenn.edu/oral-presentations-program/#oral1>  

# 1. Model Predictive Control via On-Policy Imitation Learning 
Source : <https://proceedings.mlr.press/v211/ahn23a/ahn23a.pdf>

The first  paper deals with finding an efficient way of finding a controller for constrained linear systems. The presenters choose MPC as it is able to stabilize a constrained linear system as opposed to an LQR controller. MPC is also useful in many applications, has inherent robustness and works well in practice. The idea of MPC is  
 
The limitation is that we have to solve an optimization problem at each iteration which becomes very complicated for a higher dimensional system. One can use explicit MPC to overcome this issue, but it is very hard to precompute and store. To overcome this problem, learning approach is used.
The paper introduces an interesting idea of imitation learning on Model Predictive Control. The goal is to learn the expert policy/controller based on its demonstrations. This could be a human demo as well.  We can use behavior cloning to learn the states along the expert trajectory. However, in higher dimensional cases even small approximation errors could lead to distribution shift.  To solve this problem, we use a time varying controller and apply Forward Training Algorithm where instead of learning the policy, we learn on trajectory of the expert.  
 
The drawback of using Forward Training Algorithm is that the number of stages increases with T. To make these number of stages finite, we use a property of the MPC which helps in terminating the iterations. The property is that a region near the origin, called as the positive invariance region, the MPC policy is same as the LQR policy. So whenever a we reach a state in the positive invariance region, we know that policy is going to follow LQR thereafter. Hence we can terminate the policy easily from there.
Another theorem coming out of the paper is as follows

 


In our opinion, the presentation properly addresses the issues with each controller type, each approach. After that a method to handle the drawback of each approach is mentioned which is quite helpful.  The use of a learning approach to solve controllers is very helpful for dynamic applications. Furthermore, it is interesting to see the final approach circle back to the computation of an LQR which was rejected at the very start. 
The presenters mention in the paper that an alternative approach to their method would be to learn the value function of the policy. We could use the value of each expert demonstration to improve the performance of imitation learning algorithms. It would be interesting to see if properties of MPC value function namely being convex and piecewise quadratic make learning value functions better.  
There could also be a method to combine the above method directly with the policy optimization approach.

# 2. Policy Learning for Active Target Tracking over Continuous Trajectories

Planning the trajectory of a sensing robot to reduce the uncertainty in the state of the dynamic target of interest which has been motivated by diverse applications such as wildfire detection, security and surveillance, passage envision and so on. 
Traditional active target tracking problem has been studied by the method of optimization the past planning-based techniques this work considers how to run the control policy for active target tracking that generalizes to the different target configurations and motions.  Active target tracking trajectory of a sensing agent. 
Prior to the paper, there was no policy to run a control policy to maximize the log determinant of the information matrix at the terminal time. Before the introduction of the model outlined in this paper, active target tracking faced a multitude of challenges. Uncertainty in target state estimation, influenced by both target motion and sensor observations, posed a significant hurdle. Agents equipped with sensors featuring a limited field of view (FOV) struggled to get accurate sensor measurements, decreasing their ability to gather precise information about the target's state. Considering the constraints of the FOV, planning trajectories for these agents, was a complex task. Existing approaches were often tailored to specific environments and lacked adaptability to new or dynamically changing scenarios. Achieving stable policy convergence for low-level continuous control in SE(3) agent kinematics using model-free reinforcement learning methods proved challenging. Multi-target tracking introduced further complexity, necessitating efficient resource allocation across different targets. Furthermore, learning control policies across various environments for optimal information acquisition in Simultaneous Localization and Mapping (SLAM) was non-trivial, and striking the right balance between exploration and exploitation in dynamic tracking scenarios remained a key challenge.
The trajectory planning of the sensing robot to choose optimal path impacts the accuracy of the targets. This can also be used by minimizing uncertainty in dynamic target sites. For example, wildfire detection. 
Proposed deep view network which is model-free running method over discrete action space. Proximal policy optimisation over continuous action space model-free model. Proposes model-based learning method for dynamic target in continuous action space.  We consider single-agent multiple targets, where a single agent follows a series motion model for the robot post stage which includes both the position and orientation states for a given velocity input. For each target, follow the linear Gaussian model. The onboard sensor on the mobile robot returns to the sensing stage for the target state in a limited view. Using sensor states we employ a target estimator by the Kalman filter

 ![image](https://github.com/YashasShetty/summary_of_few_l4dc_23_talks/assets/112819834/e95aa5e5-11c9-4f3e-85cf-7ef8f3b74b6b)


Equations of Kalman filter Target estimation which have prediction state and updated state (only for a target within the field of view). 
 
![image](https://github.com/YashasShetty/summary_of_few_l4dc_23_talks/assets/112819834/ee5be859-5a21-415c-a166-a668f0acc687)

![image](https://github.com/YashasShetty/summary_of_few_l4dc_23_talks/assets/112819834/3aff0574-5fc0-411b-a512-d80dc28a0d72)

The distance d(q,f) is negative because the set F or if Q is in the field of view returns a positive distance.

probit function Is approximating a discontinuous step function by a continuous smooth function in particular gaussian CDF.  Unobserved region to observed region from 0 to 1.

Model-based reinforcement learning algorithm to obtain optimal policy parameter given initial pp.
RL forward path is robot pose stare, predicted mean and information in Kalman filter in differential field of view. After calculating the gradient parameter using the terminal condition of the sensitivity variable. The analytical gradient of the reward function is given by a trace operator w.r.t sensitivity variable by taking the gradient of the information updates and robot ss3 model.
They trained the two control policies (Model-based RL and Model-free RL). Training is done on randomized Initial conditions by a randomised number of targets and randomized buyers in the target motion.  This is tested on 3,5,7 number of targets. Observed the results, it efficiently observes compared to others. Have a larger mean and smaller standard deviation in the absolute reward after sufficient episodes. As it has multiple targets, it has to do multiple tracking, priority is based on the uncertainty of each target done by the information matrix. The higher the uncertainty, the priority increases. 
The paper's contributions align with active target tracking for robotics. Its focus is on practical applications like wildfire detection and security, along with the adaptable control policy. The use of deep view networks and model-free methods speaks to your interest in flexible solutions for various action spaces.  The use of Kalman filter-based target estimation and the probit function for smooth transitions between observed and unobserved regions further solidify the paper's technical depth. It also means the data and testing it with random data points of the random variable. Overall, the paper's technical contributions strongly align with your research goals in advancing effective tracking techniques.
The open challenges after this paper are to develop RL methods for constant environments with obstacle models FOV and collision avoidance. General target dynamics, applying the conformal prediction to guarantee the probabilistic bound in the unknown target states.  Multi-agent RL where we use Graphical Neural Networks (GNNs). 


# 4. Agile Catching with Whole-Body MPC and Blackbox Policy Learning 
Source <https://sites.google.com/view/agile-catching>, <https://arxiv.org/pdf/2306.08205.pdf>

The first talk studies the relative merits of using an MPC vs Blackbox policy learning for a high-speed robotic catching application. The known advantages of an MPC are zero-shot learning, no distribution. Although it might be sensitive to model errors and is computationally complex. On the other hand, Blackbox policy learning does not need accurate dynamics and its policy inference is quite efficient. However, it is prone to distribution shift and is data insensitive.  Analyses of both these methods are presented.
Firstly, the MPC method is explained. Catching is formulated as a free-end-time constrained control trajectory norm minimization optimal control problem(OCP). 

![image](https://github.com/YashasShetty/summary_of_few_l4dc_23_talks/assets/112819834/c39579f0-5ef6-4d2d-b177-cf911dbf17e5)
 
  Ball catching is determined by the below two terms,
1. The position of the ball and center of the net are in proximity.
2. The net is aligned to the velocity of the ball at the end (i.e., time of catching). 

This catching problem is simplified to a multistage OCP for computational efficiency. The detected ball trajectory is split into multiple stages of variable time interval such that each stage can be seen as a constant acceleration stage followed by a cruise phase (zero acceleration) stage. This multi stage constrained OCP is solved via a Sequential Quadratic Programming(SQP) solver. And once the OCP is solved, the ball trajectory can be intercepted. Then an open loop cradling primitive is defined to stably catch the ball. The net is aligned with velocity of the ball while decelerating slowly. 


The blackbox(BB) policy learning method is explained next.  
The blackbox algorithm works by optimizing the weights of a policy network which takes in robot observations and generates robot commands. The policy weights are updated using an evolutiobnary startergy (es algorithm) which generates gauddian perturbations to current weight estimates and evaluates the perturbed weights on the catching task which is evaluated using a  scalar reward The reward signals are observed by the es algo and a zero order gradient estimate is calculated and used to update the weights of the policy network.

A representation of the policy architecture is show below


There are two types of considerations to the policy, 
1. Joint position history
2. Predicted ball trajectory
These considerations are processes through two CNNs and their outputs are concatenated and post processed by fully connected layers to produce joint velocity commands. 


Reward Design: The following terms are combined to formulate a scalar  reward function. Note that different terms are used in simulation and reality.
1. Object Orientation Reward (sim)
2. Object stability Reward (sim)
3. Object Position Reward (sim/real)
4. Penalties for exceeding dynamic constraints (sim/real)
5. Object catch reward (real) 

The setup 

 ![image](https://github.com/YashasShetty/summary_of_few_l4dc_23_talks/assets/112819834/24a5524e-0774-40be-bbca-32620e2075e9)


The results of the two methods are shown below.

BB catch success ~ 86%   	SQP catch success ~ 79%

BB inference time ~ 7ms   SQP solve time ~ 43ms 

![image](https://github.com/YashasShetty/summary_of_few_l4dc_23_talks/assets/112819834/69ca7ebd-ff63-4ecc-a97c-d20e603673be) 
![image](https://github.com/YashasShetty/summary_of_few_l4dc_23_talks/assets/112819834/3b48f190-7da0-4ff1-8d87-d50be200f7e7)

We can see that while SQP has a constant performance, BB performance varies with the iterations and gets better as the number of iterations increase. It surpasses the performance of SQP after a point. However, these number required iterations is undetermined and can be change when we vary environment and/or change the application.

![image](https://github.com/YashasShetty/summary_of_few_l4dc_23_talks/assets/112819834/252ad1c7-22cb-4dfa-ab8a-553a9afde1e7)

It is interesting to note that BB catches the ball on the right side, while SQP has multi modal catching ability. Which is why SQP can catch hand throws and BB fails at that. 

![image](https://github.com/YashasShetty/summary_of_few_l4dc_23_talks/assets/112819834/1e12d8fd-854b-4a2d-bee4-26524c8134e5)

We can see that SQP is robust across all speeds whereas BB gives good performance at higher speeds. When we change the yaw of the thrower, the catch success is pretty much the same for both methods. 

Links to videos
SQP: <https://www.youtube.com/watch?v=FAHLIHUf8Tc>
BB: <https://www.youtube.com/watch?v=QilzQcB-BNU>

The presentation concludes with observation that BB is a better method when it comes to catching success of mechanical throws, however SQP can handle distribution shift significantly better. Hence a possible future study would be to combine these two methods, combine the best of them to get a fused algorithm which is robust than both of them. This work can also be extended to handling of more complex objects like non-spherical objects, wiffle balls. We can also extend in the direction of multi object catching too.


# Conclusion
The above talks show us various aspects of policy learning, how we can use them with existing methods like LQR, MPC and how we can improve upon these traditional methods by combining all of their desired proporties , how we can use them for applications like target tracking, how the stand when compared to methods like the Sequential Quadratic Programming solver which gives us a good idea of how tradional methods and modern methods can turn out to be complementary. Each talk leaves us with a possibility of improvement, new expiremental ideas and results which can furthen the exising knowledge we have on controllers and policy learning methods.
