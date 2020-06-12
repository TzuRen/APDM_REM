import codecs

rf1=codecs.open('./2007_train.txt','r')
#rf1=codecs.open('./2012_train.txt','r')

wf=codecs.open('VOC_0712_train.txt','a')

lines=rf1.readlines()

for line in lines:
    wf.write(line)

