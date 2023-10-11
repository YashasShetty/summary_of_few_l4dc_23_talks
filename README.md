# summary_of_few_l4dc_23_talks

Rohit Reddy and Yashas Shetty
We are going to summarize the following topics from the l4dc 2023 conference 

1. Model Predictive Control via On-Policy Imitation Learning 
2. Policy Learning for Active Target Tracking over Continuous Trajectories 
3. Policy Evaluation in Distributional LQR 
4. Agile Catching with Whole-Body MPC and Blackbox Policy Learning 

Source :<https://l4dc.seas.upenn.edu/oral-presentations-program/#oral1>  


The common theme of these topics is policy learning. We explored different types of policy learning presented in the conference.

1. Model Predictive Control via On-Policy Imitation Learning 
Source : <https://proceedings.mlr.press/v211/ahn23a/ahn23a.pdf>
The first  paper deals with finding an efficient way of finding a controller for constrained linear systems. The presenters choose MPC as it is able to stabilize a constrained linear system as opposed to an LQR controller. MPC is also useful in many applications, has inherent robustness and works well in practice. The idea of MPC is  
 
The limitation is that we have to solve an optimization problem at each iteration which becomes very complicated for a higher dimensional system. One can use explicit MPC to overcome this issue, but it is very hard to precompute and store. To overcome this problem, learning approach is used.
The paper introduces an interesting idea of imitation learning on Model Predictive Control. The goal is to learn the expert policy/controller based on its demonstrations. This could be a human demo as well.  We can use behavior cloning to learn the states along the expert trajectory. However, in higher dimensional cases even small approximation errors could lead to distribution shift.  To solve this problem, we use a time varying controller and apply Forward Training Algorithm where instead of learning the policy, we learn on trajectory of the expert.  
 
The drawback of using Forward Training Algorithm is that the number of stages increases with T. To make these number of stages finite, we use a property of the MPC which helps in terminating the iterations. The property is that a region near the origin, called as the positive invariance region, the MPC policy is same as the LQR policy. So whenever a we reach a state in the positive invariance region, we know that policy is going to follow LQR thereafter. Hence we can terminate the policy easily from there.
Another theorem coming out of the paper is as follows

 


In our opinion, the presentation properly addresses the issues with each controller type, each approach. After that a method to handle the drawback of each approach is mentioned which is quite helpful.  The use of a learning approach to solve controllers is very helpful for dynamic applications. Furthermore, it is interesting to see the final approach circle back to the computation of an LQR which was rejected at the very start. 
The presenters mention in the paper that an alternative approach to their method would be to learn the value function of the policy. We could use the value of each expert demonstration to improve the performance of imitation learning algorithms. It would be interesting to see if properties of MPC value function namely being convex and piecewise quadratic make learning value functions better.  
There could also be a method to combine the above method directly with the policy optimization approach.




4. Agile Catching with Whole-Body MPC and Blackbox Policy Learning 
Source <https://sites.google.com/view/agile-catching>, <https://arxiv.org/pdf/2306.08205.pdf>

The first talk studies the relative merits of using an MPC vs Blackbox policy learning for a high-speed robotic catching application. The known advantages of an MPC are zero-shot learning, no distribution. Although it might be sensitive to model errors and is computationally complex. On the other hand, Blackbox policy learning does not need accurate dynamics and its policy inference is quite efficient. However, it is prone to distribution shift and is data insensitive.  Analyses of both these methods are presented.
Firstly, the MPC method is explained. Catching is formulated as a free-end-time constrained control trajectory norm minimization optimal control problem(OCP). 
 
  Ball catching is determined by the below two terms,
1. The position of the ball and center of the net are in proximity.
2. The net is aligned to the velocity of the ball at the end (i.e., time of catching). 
This catching problem is simplified to a multistage OCP for computational efficiency. The detected ball trajectory is split into multiple stages of variable time interval such that each stage can be seen as a constant acceleration stage followed by a cruise phase (zero acceleration) stage. This multi stage constrained OCP is solved via am o Sequential Quadratic Programming solver. And once the OCP is solved, the ball trajectory can be intercepted. Then an open loop cradling primitive is defined to stably catch the ball. The net is aligned with velocity of the ball while decelerating slowly. 
The blackbox policy learning method is explained next.  
The blackbox algorithm works by optimizing the weights of a policy network which takes in robot observations and generates robot commands. The policy weights are updated using an evolutiobnary startergy (es algorithm) which generates gauddian perturbations to current weight estimates and evaluates the perturbed weights on the catching task which is evaluated using a  scalar reward The reward signals are observed by the es algo and a zero order gradient estimate is calculated and used to update the weights of the policy network. A
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
 

The results of the two methods are shown below.
BB catch success ~ 86%   	SQP catch success ~ 79%
BB inference time ~ 7ms      	SQP solve time ~ 43ms 
 
We can see that while SQP has a constant performance, BB performance varies with the iterations and gets better as the number of iterations increase. It surpasses the performance of SQP after a point. However, these number required iterations is undetermined and can be change when we vary environment and/or change the application.
  
It is interesting to note that BB catches the ball on the right side, while SQP has multi modal catching ability. Which is why SQP can catch hand throws and BB fails at that.
 
We can see that SQP is robust across all speeds whereas BB gives good performance at higher speeds. When we change the yaw of the thrower, the catch success is pretty much the same for both methods. 

Links to videos
SQP: <https://www.youtube.com/watch?v=FAHLIHUf8Tc>
BB: <https://www.youtube.com/watch?v=QilzQcB-BNU>

The presentation concludes with observation that BB is a better method when it comes to catching success of mechanical throws, however SQP can handle distribution shift significantly better. Hence a possible future study would be to combine these two methods, combine the best of them to get a fused algorithm which is robust than both of them. This work can also be extended to handling of more complex objects like non-spherical objects, wiffle balls. We can also extend in the direction of multi object catching too.

