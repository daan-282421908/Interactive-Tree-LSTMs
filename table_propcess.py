import os
import csv

if __name__ == '__main__' :

    path_ = os.path.abspath('.')
    a2_path = path_ + '/a2'
    target_path = path_ + '/a2_table'

    for root,_,fnames in os.walk(a2_file):
        for fname in fnames :

            f_path = os.path.join(a2_file,fname)
            write_path = os.path.join(target_path,fname.replace('.a2','.csv'))
            csvfile = open(write_path,'wb')
            writer = csv.writer(csvfile)
            writer.writerow(['event_id','event_type','src','relation','dst'])
            csvfile.close()

            csvfile = open(write_path,'ab+')
            writer = csv.writer(csvfile)

            file_a2 = open(f_path,'r')
            line = file_a2.readline()

            line = line.replace('\n','')
            line = line.replace('\t',' ')
            line_list = line.split(' ')

            e_id = line_list[0]
            e_type = line_list[1].split(':')[0]
            sc = line_list[1].split(':')[1]

            for i in range(2,len(line_list)):
                rlt = line_list[i].split(':')[0]
                dt = line_list[i].split(':')[1]
                writer.writerow([e_id,e_type,sc,rlt,dt])
                line = file_a2.read()

        file.close()
