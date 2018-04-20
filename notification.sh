#!/bin/sh

sar -u 1 | awk '{ if (int($4)<=30) { 
                 i=i+1}
               if (i>=100) {
                      print "Sending email";
                      cmd="sendmail doctor.ahmad89@gmail.com <email.txt";
                      system(cmd);
		      exit 1;;
               }
            }'
