# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:49:02 2026

@author: MMH_user
"""

###############################################################################
def Find_My_File(filename: str, 
                 server_hard_disk_path: str = r"c:\Users\MMH_user\Desktop") -> str | None:
    
    """
    
    ABOUT:
        
     - finds file of name "filename" anywhere in "ServerHardDiscPath" 
       and returns complete path if file does exist
     - returns None and not found statement if file does not exist
     - important: returns first match only in case there are multiple files 
       with the same name!
     
     - check:
         help(Find_My_File)
         
     
     USAGE:
         
         Path = Find_My_File('12 Git and Github.pdf')
         
         Path = Find_My_File('Some Nonsense')
         
         Some Nonsense not found!
         Check spelling and/or path!
         
    """
    
    for root, _, files in os.walk(server_hard_disk_path):
        if filename in files:
            return os.path.join(root, filename)
        
    print(f"{filename} not found!\nCheck spelling and/or path!")
    return None