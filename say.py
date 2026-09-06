import cowsay
import sys
from sayings import goodbye
import pyttsx3

"""if len(sys.argv)==2:
    goodbye(sys.argv[1])"""

"""if len(sys.argv)==2:
    cowsay.trex("hello, "+sys.argv[1])"""

engine = pyttsx3.init()
this = input("What's this?")
cowsay.cow(this)
engine.say(this)
engine.runAndWait()