textFile=open("newLineCode/ch2.txt")

targetFile="newLineCode/MIT_DL_ch2_p.txt"

result=open(targetFile,"w")
targetFile_contents = textFile.read()

line =targetFile_contents.split("\n")

temp=0
for str in line:
    # result.write("\n")
    temp=temp+1
    if(temp==3):
        result.write(str+"\n"+"\n");
        temp=0
    else:
        result.write(str+"\n");
    

    