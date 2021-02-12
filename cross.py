from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import random
import numpy       
import json
import psycopg2
from functools import reduce

# Database Connection Constants 
PSQL_HOST = "ec2-52-5-176-53.compute-1.amazonaws.com"
PSQL_PORT = "5432"
PSQL_USER = "vsemigwrofvgsy"
PSQL_PASS = "dec78e50cb74ebc76535f8d151c84df463eeeaed42063d5eb51a0b2b36f81643"
PSQL_DB = "de220cmq22267r"

# Connection to DB
connection_address= """
host=%s port=%s user=%s password=%s dbname=%s
""" % (PSQL_HOST, PSQL_PORT, PSQL_USER, PSQL_PASS, PSQL_DB)
connection = psycopg2.connect(connection_address)

cursor = connection.cursor()

id_cruce=sys.argv[1]


### TRAINING CODE
def dec_a_base3(decimal): # decimal to base 3 number function
    num_base3 = ''
    while decimal // 3 != 0:
        num_base3 = str(decimal % 3) + num_base3
        decimal = decimal // 3
    return str(decimal) + num_base3

def dec_a_base2(decimal): # decimal to base 2 number function
    num_base2 = ''
    while decimal // 2 != 0:
        num_base2 = str(decimal % 2) + num_base2
        decimal = decimal // 2
    return str(decimal) + num_base2

state_m = numpy.array([[0,0,0,0,0,0,0,0]])
# Initial states array is 8 bits, because the cross has 4 directions West-East, East-West, North-South, South-North, 
# As it has 4 directions, it has 4 traffic lights and 4 vehicle queues
# 4 MSB (Most Significant Bits) of 8 bits are Times of Green in each traffic light (0 Short Time, 1 Long Time)
# 4 LSB (Least Significant Bits) of 8 bits are vehicle queues in each direction (0 Short queue, 1 Long queue)
  
# The next code adds the 256 values ??(2 to the power 8) that the states array can have
for i in range(1, 256):
   state_l= [int(x) for x in list('{0:0b}'.format(i))]
   state_a= numpy.array(state_l)
   if state_a.size < 8:
     for i in range(state_a.size,8):
       state_a = numpy.append([0], state_a)
   state_m = numpy.append(state_m, [state_a], axis=0)

action_m = numpy.array([[0,0,0,0]])
# Initial action array is 4 digits base 3 number (0, 1 or 2), one action for each direction 
# Each digit in action has value 0,1, or 2. 
# The value 0 in action is when the same time is kept in green, 
# the value 1 in action is when the time in green is decreased,
# the value of 2 in the action is when the time in green is increased

# The next code adds the 81 values ??(3 to the power 4) that the action array can have
for i in range(1, 81):
   action_l = dec_a_base3(i)
   action_l = [int(x) for x in str(action_l)]
   action_a= numpy.array(action_l)
   if action_a.size < 4:
     for i in range(action_a.size,4):
       action_a = numpy.append([0], action_a)
   action_m = numpy.append(action_m, [action_a], axis=0)

# Initial values (array of zeros) of next_state_matrix and reward_matrix
next_state_matrixP = numpy.zeros((256, 81))
reward_matrixP = numpy.zeros((256, 81))

for i in range(0, 256):
  for j in range(0, 81):
     reward= numpy.zeros(4)
     next_state_response = numpy.zeros(8)
          
     for k in range(0, 4):
       if ((state_m[i,k] == 0) and (state_m[i,k+4] == 0) and (action_m[j,k] == 0)):
           # If the Green Time is short, the car queue is short and the action is to keep the Green Time is something logical
           reward[k] = 5      
           next_state_response[k] = 0
           next_state_response[k+4] = 0
       if (state_m[i,k] == 0) and (state_m[i,k+4] == 0) and (action_m[j,k] == 1):
           # If the Green Time is short, the car queue is short and the action is to decrease the Green Time is something illogical, it cannot be decreased any more
           reward[k] = 0    
           next_state_response[k] = 0
           next_state_response[k+4] = 0
       if (state_m[i,k] == 0) and (state_m[i,k+4] == 0) and (action_m[j,k] == 2):
           # If the Green Time is short, the queue for cars is short and the action is to increase the Green Time is something illogical, it would not be required to increase
           reward[k] = 0    
           next_state_response[k] = 1
           next_state_response[k+4] = 0
       
       if (state_m[i,k] == 0) and (state_m[i,k+4] == 1) and (action_m[j,k] == 0):
           # If the Green Time is short, the queue from cars is long and the action is to keep the Green Time is not very useful.
           reward[k] = 1    
           next_state_response[k] = 0
           next_state_response[k+4] = 1
       if (state_m[i,k] == 0) and (state_m[i,k+4] == 1) and (action_m[j,k] == 1):
           # If the Time in Green is short, the queue from cars is long and the action is to decrease the Time in Green is something illogical, it cannot be decreased further
           reward[k] = 1    
           next_state_response[k] = 0
           next_state_response[k+4] = 1
       if (state_m[i,k] == 0) and (state_m[i,k+4] == 1) and (action_m[j,k] == 2):
           # If the Green Time is short, the queue from cars is long and the action is to increase the Green Time is something logical and useful.
           reward[k] = 10    
           next_state_response[k] = 1
           next_state_response[k+4] = 1

       if (state_m[i,k] == 1) and (state_m[i,k+4] == 0) and (action_m[j,k] == 0):
           # If the Green Time is long, the queue for cars is short and the action is to keep the Green Time could be useless
           reward[k] = 1    
           next_state_response[k] = 1
           next_state_response[k+4] = 0
       if (state_m[i,k] == 1) and (state_m[i,k+4] == 0) and (action_m[j,k] == 1):
           # If the Green Time is long, the queue for cars is short and the action is to decrease the Green Time is something logical and can be useful.
           reward[k] = 5    
           next_state_response[k] = 0
           next_state_response[k+4] = 0
       if (state_m[i,k] == 1) and (state_m[i,k+4] == 0) and (action_m[j,k] == 2):
           # If the Green Time is long, the queue for cars is short and the action is to increase the Green Time is something illogical.
           reward[k] = 0    
           next_state_response[k] = 1
           next_state_response[k+4] = 0
           
       if (state_m[i,k] == 1) and (state_m[i,k+4] == 1) and (action_m[j,k] == 0):
           # If the Green Time is long, the car queue is long and the action is to keep the Green Time is very logical
           reward[k] = 10    
           next_state_response[k] = 1
           next_state_response[k+4] = 1
       if (state_m[i,k] == 1) and (state_m[i,k+4] == 1) and (action_m[j,k] == 1):
           # If the Time in Green is long, the queue for cars is long and the action is to decrease the Time in Green is something illogical.
           reward[k] = 0    
           next_state_response[k] = 0
           next_state_response[k+4] = 1
       if (state_m[i,k] == 1) and (state_m[i,k+4] == 1) and (action_m[j,k] == 2):
           # If the Time in Green is long, the queue for cars is long and the action is to increase the Time in Green, it is something illogical, it cannot be increased more.
           reward[k] = 0    
           next_state_response[k] = 1
           next_state_response[k+4] = 1
     
     final_reward = reward[0]+reward[1]+reward[2]+reward[3]
     reward_matrixP[i,j] = final_reward
     # The following is to go from a binary array to an integer value
     next_state_value = reduce(lambda a,b: 2*a+b, next_state_response)
     next_state_matrixP[i,j] =next_state_value

# So far the matrix P (reward) has been calculated, now we proceed to fill the matrix Q with training values ("Training")

# Hyperparameters
alpha = 0.6
gamma = 0.4
epsilon = 0.9
matrix_Q = numpy.zeros((256, 81))
matrix_Q = numpy.loadtxt('q_table.txt') # load previus values from q_table

### END OF TRAINING CODE

### RECOMMENDATION FUNCTION
def recomendacion_estado_siguiente(state):
    action = numpy.argmax(matrix_Q[state])
    recommendation = dec_a_base3(action)
    return recommendation
### RECOMMENDATION FUNCTION END

### ACTION TO STL PLAN FUNCTION
def time_converter(matrix_Q,matrix_P):
    next_state_temp=[]
    traffic_light = ""
    matrix_P=[int(i) for i in matrix_P]

    for i in range (0,4):
        if (matrix_Q[i] ==0 and matrix_P[i]==0): 
            next_state_temp.append(0)
        if (matrix_Q[i] ==1 and matrix_P[i]==0): 
            next_state_temp.append(1)

        if (matrix_Q[i] ==0 and matrix_P[i]==1): 
            next_state_temp.append(0)
        if (matrix_Q[i] ==1 and matrix_P[i]==1): 
            next_state_temp.append(0)

        if (matrix_Q[i] ==0 and matrix_P[i]==2): 
            next_state_temp.append(1)
        if (matrix_Q[i] ==1 and matrix_P[i]==2): 
            next_state_temp.append(1)
    
    print("<table width='35%'  align=center cellpadding=5 border=1 bgcolor='#2E4053'>")
    print("<tr><td bgcolor='#F4B120' align=center> EastWest ")
    print("</td>")
    print("<td bgcolor='#F4B120' align=center> SouthNort ")
    print("</td>")
    print("<td bgcolor='#F4B120' align=center> WestEast ")
    print("</td>")
    print("<td bgcolor='#F4B120' align=center> NortSouth ")
    print("</td></tr>")
    for i in range (0,4):
        if next_state_temp[i]==0:
            print("<td bgcolor='#EEEEEE' align=center> ShortTime </td>")
            traffic_light=traffic_light+"ST"
        if next_state_temp[i]==1:
            print("<td bgcolor='#EEEEEE' align=center> LongTime </td>")
            traffic_light=traffic_light+"LT"
    print("</table>")
    return traffic_light,next_state_temp
	# LT means Long Time, Long time of traffic_light, ST means Short Time, Short time of
### ACTION TO STL PLAN FUNCTION END

### CODE FOR RECOMMENDATION
current_state = [0,0,0,0,0,0,0,0]

sql2 = "select plan from cruce where id='"+id_cruce+"';"
cursor.execute(sql2)
rows = cursor.fetchall()
for row in rows:
    plan = row[0]

print("<center> <h3>Current traffic light time</h3>")
print("<table width='35%'  align=center cellpadding=5 border=1 bgcolor='#2E4053'>")
print("<tr><td bgcolor='#F4B120' align=center> EastWest")
print("</td>")
print("<td bgcolor='#F4B120' align=center> SouthNort")
print("</td>")
print("<td bgcolor='#F4B120' align=center> WestEast")
print("</td>")
print("<td bgcolor='#F4B120' align=center> NortSouth")
print("</td></tr><tr>")

if(plan[0:2]=="ST"):
    current_state[0] = 0
    print("<td bgcolor='#EEEEEE' align=center> ShortTime")
    print("</td>")

else:
    current_state[0] = 1
    print("<td bgcolor='#EEEEEE' align=center> LongTime")
    print("</td>")

if(plan[2:4]=="ST"):
    current_state[1] = 0
    print("<td bgcolor='#EEEEEE' align=center> ShortTime")
    print("</td>")

else:
    current_state[1] = 1
    print("<td bgcolor='#EEEEEE' align=center> LongTime")
    print("</td>")

if(plan[4:6]=="ST"):
    current_state[2] = 0
    print("<td bgcolor='#EEEEEE' align=center> ShortTime")
    print("</td>")

else:
    current_state[2] = 1
    print("<td bgcolor='#EEEEEE' align=center> LongTime")
    print("</td>")

if(plan[6:8]=="ST"):
    current_state[3] = 0
    print("<td bgcolor='#EEEEEE' align=center> ShortTime")
    print("</td>")

else:
    current_state[3] = 1
    print("<td bgcolor='#EEEEEE' align=center> LongTime")
    print("</td>")

print("</tr></table>")
# THIS IS THE ORDER OF PRIORITY OF THE TRAFFIC LIGHT, FIRST IT TURNS GREEN EW, THEN SN, THEN WE AND FINALLY NS.
sql1 = "SELECT * FROM colas_reales where id_cruce='"+id_cruce+"';"
cursor.execute(sql1)
queues = cursor.fetchall()
for queu in queues:
    print("<br> <h3> Queues obtained </h3>")

queuEW = queu[2]
queuSN = queu[3]
queuWE = queu[4]
queuNS = queu[5]
print("<table width='35%'  align=center cellpadding=5 border=1 bgcolor='#2E4053'>")
print("<tr><td bgcolor='#EEEEEE' align=center> EastWest: ", queuEW)
print("</td>")
print("<td bgcolor='#EEEEEE' align=center> SouthNort: ", queuSN)
print("</td>")
print("<td bgcolor='#EEEEEE' align=center> WestEast: ", queuWE)
print("</td>")
print("<td bgcolor='#EEEEEE' align=center> NortSouth: ", queuNS)
print("</td></tr> </table>")

sql = "SELECT ew, sn, we, ns FROM cruce where id='"+id_cruce+"';"
cursor.execute(sql)
parameters = cursor.fetchall()
for parameter in parameters:
    print("<br>")
        
if queuEW > parameter[0]:
    current_state[4]=1
else:    
    current_state[4]=0
if queuSN > parameter[1]:
    current_state[5]=1
else:
    current_state[5]=0    
if queuWE > parameter[2]:
    current_state[6]=1
else:    
    current_state[6]=0
if queuNS > parameter[3]:
    current_state[7]=1
else:
    current_state[7]=0

state_bin=str(current_state[0])+str(current_state[1])+str(current_state[2])+str(current_state[3])+str(current_state[4])+str(current_state[5])+str(current_state[6])+str(current_state[7])
#state_bin="10101010"
state_dec=int(state_bin, base=2)
action_tt=recomendacion_estado_siguiente(state_dec)
action_tt=list(str(action_tt))
for i in range(len(action_tt),4):  
	action_tt.insert(0,0)

print("<br> <h3> New Time Traffic Lights: </h3>")
new_time, new_state=time_converter(current_state,action_tt)

for i in range(0,4):
    current_state[i]=new_state[i]

### UPDATE PLAN IN DB
print("<br><center><h3>Do you want to update the traffic light plan?</h3><form method=POST action='tabla_recomendacion.php'><table border=0 width='50%' align=center>")
print("<input type='hidden' name=plan value='"+new_time+"' readonly='readonly' required>")
print("</table><input type='hidden' value='S' name='enviado'><input type='hidden' value='"+id_cruce+"' name='id_cruce'> <input type=submit value='Update' name='Modificar'></center>")
### END OF UPDATE PLAN IN DB

cursor.close()
connection.close()
sys.stdout.flush()
### END OF CODE FOR RECOMMENDATION
