#! /c/Apps/Anaconda3/python
"""
[Title] Understanding Web Interface of Analytics
[Author] Yibeck Lee(Yibeck.Lee@gmail.com)
[Program Code Name] e200-flask-tensorflow-simple-test.py  
[Description]
  - 교육생 실습용
[History]
  - 2019-05-01 : 최초 작성
[References]
  - 
"""
from flask import Flask
import tensorflow as tf
app = Flask(__name__)

@app.route('/')
def Vector_Sum():
	sess = tf.Session()
	a = tf.constant(1)
	b = tf.constant(2)
	return str('a+b={}'.format(sess.run(a+b)))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
