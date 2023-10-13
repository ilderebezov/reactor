from src.server import app


def init():
    """
    start server
    :return:
    """
    app.run(host='127.1.1.1', debug=True, port=5000)


if __name__ == '__main__':
    init()
