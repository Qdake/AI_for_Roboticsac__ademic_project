from PIL import Image
from PIL import ImageDraw
# data
import pickle
file = open("./log_SHINE/position_record_2_sigma1_nbgen_100_sizepop_250",'rb')
data = pickle.load(file)
file.close()
# maze
maze = Image.open('./maze_hard.pbm')
maze.save('./maze_hard.png')
maze = Image.open('./maze_hard.png')
maze.show()
draw = ImageDraw.Draw(maze)
for p in data:
    draw.ellipse([(p[0],p[1]),(p[0]+2,p[1]+2)],fill = "red")