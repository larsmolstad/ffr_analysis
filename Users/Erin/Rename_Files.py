#You must first run the cookbook on the correct data range, and then you can run
#this script to save the raw data in a new folder within FFR named '_RENAMEDATA'

import polygon_utils_old as pols
import os
import shutil
import math 
##The uncomented parts below are not needed to run, but shows
##how i picked out the fields, i just used the corners of the 
##existing rectangles and made two new fields
#field0 = [rectangles[201][3],rectangles[232][2],rectangles[332][1],rectangles[301][0]]
#field1 = [rectangles[401][3],rectangles[432][2],rectangles[532][1],rectangles[501][0]]
#
r4 = [rectangles[101][0],rectangles[132][1],rectangles[232][2],rectangles[201][3]]
r5 = [rectangles[301][0],rectangles[332][1],rectangles[432][2],rectangles[401][3]]
r6 = [rectangles[501][0],rectangles[532][1],rectangles[632][2],rectangles[601][3]]

## This divides up the rectangles i created in the lines above
#r0 = pols.divide_rectangle(field0,2,True)[1]
#r1 = pols.divide_rectangle(field0,2,True)[0]
#r2 = pols.divide_rectangle(field1,2,True)[1]
#r3 = pols.divide_rectangle(field1,2,True)[0]
#

#Here are the coordinates of each corner, it is the same as is returned above
r0 = [[599219.8817494238, 6615162.794659675],
 [599218.0659625774, 6615164.7928884225],
 [599282.0832909764, 6615222.965318874],
 [599283.8990778228, 6615220.967090127]]

r1 =[[599221.6975362703, 6615160.796430928],
 [599219.8817494238, 6615162.794659675],
 [599283.8990778228, 6615220.967090127],
 [599285.7148646693, 6615218.96886138]]

r2 = [[599238.0396178886, 6615142.812372198],
 [599236.2238310421, 6615144.810600946],
 [599300.2411594411, 6615202.983031398],
 [599302.0569462876, 6615200.98480265]]

r3= [[599239.8554047351, 6615140.81414345],
 [599238.0396178886, 6615142.812372198],
 [599302.0569462876, 6615200.98480265],
 [599303.8727331341, 6615198.986573902]]

#Uncomment this if you want to show the selection fields on the plot with the dots

newrect= {1:r0,2:r1,3:r2,4:r3,5:r4,6:r5,7:r6}

rectbcp = rectangles

rectbcp[1000] =r0
rectbcp[1001] =r1
rectbcp[1002] =r2
rectbcp[1003] =r3


#Theese are the polygons which are checked if a measurement is inside, 
polys = [r0,
         r1,
         r2,
         r3,
         r4,
         r5,
         r6]


#And the specific heading which is given if a measurement is within the field
#Theese might be wrong, i just used the headings from some other measurements
headings = ["h-2_4000000000",
            "h0_720000000000",
            "h-2_4000000000",
            "h0_720000000000"]

ndf = df[np.isnan(np.array(df["heading"]))].sort_values(by="date")


def checkHeading(x,y,plotnr,filename,polys,headings): # x,y is coordinates, plotnumber is the assigned plot number
    #Filename is the given filename, polys is an array of polygons to check against, headings is an array of headings 
    #assigned to each plygon
    i = 0
    a = filename.split("_") #Splits up the filename by "_", could probably be done better but it works
#    if plotnr == -100: #If it isnt given a plot number check if it is inside the specified polygon
    if "-h" not in filename:
        for poly in polys:#Iterate through the ploygons specified
            if pols.point_inside_polygon(x, y, poly):#If it is inside the polygon
                #return the original filename and Add the heading to the new filename 
                return filename, "_".join(a[0:4])+"-"+headings[i]+"_"+"_".join(a[4:])
            i +=1
    return

def checkPos(x,y,plotnr,filename,polys,headings): # x,y is coordinates, plotnumber is the assigned plot number
    #Filename is the given filename, polys is an array of polygons to check against, headings is an array of headings 
    #assigned to each plygon
    i = 0
    for poly in polys:#Iterate through the ploygons specified
        if pols.point_inside_polygon(x, y, poly):#If it is inside the polygon
            #return the original filename and Add the heading to the new filename 
            return i
        i +=1
    return -1

def addkHeading(filename,heading):
    a = filename.split("_")
    heading_string = 'h{:0<14}'.format(heading).replace(".","_")
    if "-h" not in filename:
        return filename, "_".join(a[0:4])+"-"+heading_string+"_"+"_".join(a[4:])


def copy_rename(filelist,old_dir,new_dir):
    length = len(filelist)
    i= 0
    for copy_names in filelist:
        i +=1
        full_file_name = os.path.join(old_dir, copy_names[0]) #get the full path of old filename
        full_file_name_tmp = os.path.join(new_dir, copy_names[0]) #New path with old filename
        full_file_name_new = os.path.join(new_dir, copy_names[1])# new path with the new filename to be assigned
        if (os.path.isfile(full_file_name)): #Check if it is a file where we want to copy from
            shutil.copy(full_file_name, new_dir) #Copy the file to new directory
            os.rename(full_file_name_tmp, full_file_name_new) #Rename the copied file to the new name with heading
            os.remove(full_file_name)
        if i%100 == 0:
            print (i,length)


#filelist = df.apply(lambda row:checkHeading(row['x'], row['y'],row["plot_nr"],row["filename"],polys,headings), axis=1)
pd.set_option('display.max_rows', 20)
ndf["tmp_plot"] = ndf.apply(lambda row:checkPos(row['x'], row['y'],row["plot_nr"],row["filename"],polys,headings), axis=1)
ndf["angle"] = np.arctan2(ndf["y"].diff(),ndf["x"].diff())#Get the angle from change in coordinates

#
ndf["e3"] = ndf["tmp_plot"].shift(1) #shift rows
ndf["e4"] = ndf["tmp_plot"] != ndf["e3"] #bool if same sampling nr
ndf["sampling_nr"] = ndf["e4"].cumsum() #Give each sampling nr an int

#find beadian angle of each sampling group
ndf["mean_angle"] = ndf.groupby("sampling_nr")["angle"].transform("median").round(2)

filelist = ndf.apply(lambda row:addkHeading(row["filename"],row["mean_angle"]), axis=1)

def test_nrs(df, plot_numbers):
    plot_rectangles(newrect, names=True)
    for nr in plot_numbers:
        d = df[df.plot_nr == nr]
        plt.plot(d.x, d.y, marker='.')
        

plt.cla()
test_nrs(ndf, sorted(set(ndf.plot_nr)))
plt.show()


print(ndf[["date","tmp_plot","angle","sampling_nr","mean_angle"]])#.iloc[100])

spath = "Y:\\MINA\\Miljøvitenskap\\Jord\\" #the base folder path

#spath = "C:\\Users\\FredrikNerol\\Dropbox\\universitetsjobb\\peter dørch\\robot program\\"

new_dir = spath+'FFR\\_RENAMEDATA'
old_dir = spath+'FFR\\_RAWDATA'


#Uncomment to rename
copy_rename(filelist,old_dir,new_dir)
