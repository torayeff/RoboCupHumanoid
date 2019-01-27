import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--root', default='data/image_tags/', help='root folder')
parser.add_argument('--oldFile', required=True, help='name of the old file')
parser.add_argument('--yoloLabels', required=True, help='name of the the yolo labels')
parser.add_argument('--newFile', default='newFile.txt', help='name of the the new file with merged labels')

opt = parser.parse_args()

root = opt.root 

i = 0
j = 0
with open(root + opt.newFile, "w") as newFile: 
	with open(root + opt.oldFile) as final:
		with open(root + opt.yoloLabels) as yolo:
			flag = 0
			for line in final:
				yolo.seek(0)
				if line.startswith("label::ball") or line.startswith("ball"):
					i += 1
					pieces = line.split('|')
					if 'not_in_image' in line:
						img_name = pieces[1]
						for row in yolo:
							j += 1

							if img_name in row and 'not in image' not in row: # The image has a detection from yolo
								new_line = row
								newFile.write(new_line)
								break

					# if flag == 0:
					# 	new_line = pieces[1] + "|" + "ball|"
					# 	if len(pieces) <= 3:
					# 		new_line += "not in image|"
					# 	else:
					# 		new_line += "{\"x1\":\"" + pieces[4] + "\",\"y1\":\"" + pieces[5] + "\",\"x2\":\"" + pieces[6] + "\",\"y2\":\"" + pieces[7] + "\"}|"

					# 	newFile.write(new_line + "\n")
					# elif flag == 1:
					# 	newFile.write(new_line + "\n")
					# 	flag = 0
print("New file is created: " + root + opt.newFile)
print(i, j)