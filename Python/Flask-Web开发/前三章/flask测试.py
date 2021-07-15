from flask import Flask, app,render_template
from flask import make_response

app = Flask('flask_app')

@app.route('/')
def index():
    return '<h1>hello world!<h1>'

# 动态路由并返回响应对象
@app.route('/app/<name>')
def function(name):
    response = make_response('<h1>hello {name}!<h1>'.format(name=name))
    response.set_cookie('answer','42')
    return response

@app.route('/app/render/<name>')
def render(name):
    return render_template('index.html',name=name)

if __name__ == '__main__':
    app.run()