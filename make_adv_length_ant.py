import xml.etree.ElementTree as ET 
import time
import random

tree = ET.parse('/home/takaaki/.local/lib/python3.8/site-packages/gym/envs/mujoco/assets/ant_adv.xml')


root = tree.getroot()
	
len_ratio = {}
			
name_list = ["aux_1_geom","aux_2_geom","aux_3_geom","aux_4_geom"
			,"front_left_leg_geom","front_right_leg_geom","back_left_leg_geom","back_right_leg_geom"
			,"front_left_ankle_geom","front_right_ankle_geom","back_left_ankle_geom","back_right_ankle_geom"]

fromto = 	{"aux_1_geom":[0.0,0.0,0.0,0.2,0.2,0.0],"aux_2_geom":[0.0,0.0,0.0,-0.2,0.2,0.0]
			,"aux_3_geom":[0.0,0.0,0.0,-0.2,-0.2,0.0],"aux_4_geom":[0.0,0.0,0.0,0.2,-0.2,0.0]
			,"front_left_leg_geom":[0.0,0.0,0.0,0.2,0.2,0.0],"front_right_leg_geom":[0.0,0.0,0.0,-0.2,0.2,0.0]
			,"back_left_leg_geom":[0.0,0.0,0.0,-0.2,-0.2,0.0],"back_right_leg_geom":[0.0,0.0,0.0,0.2,-0.2,0.0]
			,"front_left_ankle_geom":[0.0,0.0,0.0,0.4,0.4,0.0],"front_right_ankle_geom":[0.0,0.0,0.0,-0.4,0.4,0.0]
			,"back_left_ankle_geom":[0.0,0.0,0.0,-0.4,-0.4,0.0],"back_right_ankle_geom":[0.0,0.0,0.0,0.4,-0.4,0.0]}

body_b = {"torso":[0,0,0.75],
		"torso_1_body":[0,0,0],"torso_2_body":[0,0,0]
		,"torso_3_body":[0,0,0],"torso_4_body":[0,0,0],
		"aux_1_body":[0.2,0.2,0],"aux_2_body":[-0.2,0.2,0]
		,"aux_3_body":[-0.2,-0.2,0],"aux_4_body":[0.2,-0.2,0],
		"front_left_leg_body":[0.2,0.2,0],"front_right_leg_body":[-0.2,0.2,0],
		"back_left_leg_body":[-0.2,-0.2,0],"back_right_leg_body":[0.2,-0.2,0]} # 15
		
	
depend = {"aux_1_body":"aux_1_geom","aux_2_body":"aux_2_geom",
		"aux_3_body":"aux_3_geom","aux_4_body":"aux_4_geom",
		"front_left_leg_body":"front_left_leg_geom","front_right_leg_body":"front_right_leg_geom",
		"back_left_leg_body":"back_left_leg_geom","back_right_leg_body":"back_right_leg_geom"} 

def set_geom(geom,epsilon):
	new_geom = []
	dlt_geom = []
	
	seed = time.time() 
	r = random.random()	
	ratio = 1.0
	
	new_geom.append(geom[0])
	new_geom.append(geom[1])
	new_geom.append(geom[2])
	new_geom.append(geom[0] + (geom[3]-geom[0])*ratio)
	new_geom.append(geom[1] + (geom[4]-geom[1])*ratio)
	new_geom.append(geom[2] + (geom[5]-geom[2])*ratio)		
	
	for i in range(6):
		dlt_geom.append(new_geom[i] - geom[i])
	
	xml_geom = make_xml(new_geom)
	
	return ratio, dlt_geom, xml_geom

def set_body(body,dlt_geom):
	new_body = []
	for i in range(3):
		new_body.append(body[i] + dlt_geom[i+3])
	xml_body = make_xml(new_body)
	return xml_body

def make_xml(my_list):
	my_xml = ""
	for i in range(len(my_list)):
		my_xml += str(my_list[i])
		my_xml += " "
	return my_xml

def get_body_from_geom(depend,value):
	keys = [k for k,v in depend.items() if v == value]
	return keys

len_epsilon = 0.1

for geom in root.iter("geom"):
	for name in name_list:
		if name in str(geom.attrib):
			len_ratio[name], dlt_geom, xml_geom = set_geom(fromto.get(name),len_epsilon)#
			geom.set("fromto",xml_geom)
			
			for body in root.iter("body"):	
				tmp_depend = get_body_from_geom(depend,name)
				if len(tmp_depend) == 1:
					if tmp_depend[0] in str(body.attrib):
						xml_body = set_body(body_b[tmp_depend[0]],dlt_geom)#
						body.set("pos",xml_body)
		
tree.write('/home/takaaki/.local/lib/python3.8/site-packages/gym/envs/mujoco/assets/ant_adv.xml',encoding='UTF-8')




def set_adv_length(length_ratio):

		
	tree = ET.parse('/home/takaaki/.local/lib/python3.8/site-packages/gym/envs/mujoco/assets/ant_adv.xml')


	root = tree.getroot()
		
	len_ratio = {}
				
	name_list = ["aux_1_geom","aux_2_geom","aux_3_geom","aux_4_geom"
				,"front_left_leg_geom","front_right_leg_geom","back_left_leg_geom","back_right_leg_geom"
				,"front_left_ankle_geom","front_right_ankle_geom","back_left_ankle_geom","back_right_ankle_geom"]


	fromto = 	{"aux_1_geom":[0.0,0.0,0.0,0.2,0.2,0.0],"aux_2_geom":[0.0,0.0,0.0,-0.2,0.2,0.0]
				,"aux_3_geom":[0.0,0.0,0.0,-0.2,-0.2,0.0],"aux_4_geom":[0.0,0.0,0.0,0.2,-0.2,0.0]
				,"front_left_leg_geom":[0.0,0.0,0.0,0.2,0.2,0.0],"front_right_leg_geom":[0.0,0.0,0.0,-0.2,0.2,0.0]
				,"back_left_leg_geom":[0.0,0.0,0.0,-0.2,-0.2,0.0],"back_right_leg_geom":[0.0,0.0,0.0,0.2,-0.2,0.0]
				,"front_left_ankle_geom":[0.0,0.0,0.0,0.4,0.4,0.0],"front_right_ankle_geom":[0.0,0.0,0.0,-0.4,0.4,0.0]
				,"back_left_ankle_geom":[0.0,0.0,0.0,-0.4,-0.4,0.0],"back_right_ankle_geom":[0.0,0.0,0.0,0.4,-0.4,0.0]}

	body_b = {"torso":[0,0,0.75],
			"torso_1_body":[0,0,0],"torso_2_body":[0,0,0]
			,"torso_3_body":[0,0,0],"torso_4_body":[0,0,0],
			"aux_1_body":[0.2,0.2,0],"aux_2_body":[-0.2,0.2,0]
			,"aux_3_body":[-0.2,-0.2,0],"aux_4_body":[0.2,-0.2,0],
			"front_left_leg_body":[0.2,0.2,0],"front_right_leg_body":[-0.2,0.2,0],
			"back_left_leg_body":[-0.2,-0.2,0],"back_right_leg_body":[0.2,-0.2,0]} # 15
			
		
	depend = {"aux_1_body":"aux_1_geom","aux_2_body":"aux_2_geom",
			"aux_3_body":"aux_3_geom","aux_4_body":"aux_4_geom",
			"front_left_leg_body":"front_left_leg_geom","front_right_leg_body":"front_right_leg_geom",
			"back_left_leg_body":"back_left_leg_geom","back_right_leg_body":"back_right_leg_geom"} 

	def set_geom2(geom,epsilon,leng_ratio):
		new_geom = []
		dlt_geom = []
		
		seed = time.time() 
		r = random.random()	
		ratio = leng_ratio
		
		new_geom.append(geom[0])
		new_geom.append(geom[1])
		new_geom.append(geom[2])
		new_geom.append(geom[0] + (geom[3]-geom[0])*ratio)
		new_geom.append(geom[1] + (geom[4]-geom[1])*ratio)
		new_geom.append(geom[2] + (geom[5]-geom[2])*ratio)		
		
		for i in range(6):
			dlt_geom.append(new_geom[i] - geom[i])
		
		xml_geom = make_xml(new_geom)
		
		return ratio, dlt_geom, xml_geom

	def set_body(body,dlt_geom):
		new_body = []
		for i in range(3):
			new_body.append(body[i] + dlt_geom[i+3])
		xml_body = make_xml(new_body)
		return xml_body

	def make_xml(my_list):
		my_xml = ""
		for i in range(len(my_list)):
			my_xml += str(my_list[i])
			my_xml += " "
		return my_xml

	def get_body_from_geom(depend,value):
		keys = [k for k,v in depend.items() if v == value]
		return keys

	len_epsilon = 0.1

	length_ratio_dict = {}
	i = 0
	for name in name_list: # list=>dict
		length_ratio_dict[name] = length_ratio[i]
		i += 1

	for geom in root.iter("geom"):
		for name in name_list: #example "aux_1_geom"
			if name in str(geom.attrib):
				len_ratio[name], dlt_geom, xml_geom = set_geom2(fromto.get(name),len_epsilon,length_ratio_dict[name])#
				geom.set("fromto",xml_geom)
				
				# rgb
				length_epsilon = 0.05
				if length_ratio_dict[name] < 1: #small
					red = 0.5 + ((1-length_ratio_dict[name])/length_epsilon)*0.5
					other1 = 0.5 - ((1-length_ratio_dict[name])/length_epsilon)*0.5
					geom.set("rgba",str(red)+str(" ")+str(other1)+str(" ")+str(other1)+str(" 1")) 
				else: # big
					blue = 0.5 + ((length_ratio_dict[name]-1)/length_epsilon)*0.5
					other2  = 0.5 - ((length_ratio_dict[name]-1)/length_epsilon)*0.5
					geom.set("rgba",str(other2)+str(" ")+str(other2)+str(" ")+str(blue)+str(" 1"))
				#
				
				for body in root.iter("body"):	
					tmp_depend = get_body_from_geom(depend,name)
					if len(tmp_depend) == 1:
						if tmp_depend[0] in str(body.attrib):
							xml_body = set_body(body_b[tmp_depend[0]],dlt_geom)#
							body.set("pos",xml_body)

	tree.write('/home/takaaki/.local/lib/python3.8/site-packages/gym/envs/mujoco/assets/ant_adv.xml',encoding='UTF-8')
