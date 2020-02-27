
import numpy as np

import smtplib



def send_email(Name,metrics):
    # creates SMTP session
    s = smtplib.SMTP('smtp.gmail.com', 587)
    # start TLS for security
    s.starttls()
    # Authentication
    s.login("Hanbin1805@gmail.com", "SnowMan1314")
    # sending the mail
    message=str(Name)
    for item in metrics:
        message=message+'\n'+str(item)+'\n'+str(np.array(metrics[item]))
    # message = str(Name) + '\n' + str(np.array(mAP_val)) + '\n' + str(np.array(ACC_val))
    # terminating the session
    s.sendmail("hanbin1805@gmail.com", "305361709@qq.com", message)
    s.quit()