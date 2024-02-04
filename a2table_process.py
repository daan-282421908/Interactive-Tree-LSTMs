import os
import csv

path_ = os.path.abspath('.')


for r,_,fs in os.walk(a2_path):
    for f in fs:
        fpath = os.path.join(a2_path,f)
        new_path = os.path.join(target_path,fname.replace('.a2','.csv'))

        csv_f = open(write_path,'wb')
        writer = csv.writer(csv_f)
        writer.writerow(['modification_id','modification_type','dst'])
        csvfile.close()


        with open(write_path,'ab+') as csf_f:
            writer = csv.writer(csv_f)

        with open(f_path,'r') as fa2:
            line = fa2.readline()
        
        while line :
            if (line[0] != 'M') and (line[0] != 'A'): 
                line = file_a2.readline()
                continue

            line = line.replace('\n','')
            line = line.replace('\t',' ')
            line_list = line.split(' ')

            m_id = line_list[0]
            m_type = line_list[1]
            dt = line_list[2]
            writer.writerow([m_id,m_type,dt])
            
            line = file_a2.readline()

        file_a2.close()
        csvfile.close()
