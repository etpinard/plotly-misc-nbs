import time
import httplib
import StringIO


class Stream:
    def __init__(self, server, port=80, headers={}):
        ''' Initialize a stream object and an HTTP Connection
        with chunked Transfer-Encoding to server:port with optional headers.
        '''
        self.maxtries = 5
        self._tries = 0
        self._delay = 1
        self._closed = False
        self._server = server
        self._port = port
        self._headers = headers
        self._connect()

    def write(self, data, reconnect_on=('', 200, )):
        ''' Send `data` to the server in chunk-encoded form.
        Check the connection before writing and reconnect
        if disconnected and if the response status code is in `reconnect_on`.

        The response may either be an HTTPResponse object or an empty string.
        '''

        if not self._isconnected():

            # Attempt to get the response.
            response = self._getresponse()

            # Reconnect depending on the status code.
            if ((response == '' and '' in reconnect_on()) or
                (response and isinstance(response, httplib.HTTPResponse) and
                 response.status in reconnect_on)):
                self._reconnect()

            elif response and isinstance(response, httplib.HTTPResponse):
                # If an HTTPResponse was recieved then
                # make the users aware instead of
                # auto-reconnecting in case the
                # server is responding with an important
                # message that might prevent
                # future requests from going through,
                # like Invalid Credentials.
                # This allows the user to determine when
                # to reconnect.
                raise Exception("Server responded with "
                                "status code: {status_code}\n"
                                "and message: {msg}."
                                .format(status_code=response.status,
                                        msg=response.read()))

            elif response == '':
                raise Exception("Attempted to write but socket "
                                "was not connected.")

        try:
            msg = data
            msglen = format(len(msg), 'x')  # msg length in hex
            # Send the message in chunk-encoded form
            self._conn.send('{msglen}\r\n{msg}\r\n'
                            .format(msglen=msglen, msg=msg))
        except httplib.socket.error:
            self._reconnect()
            self.write(data)

    def _connect(self):
        ''' Initialize an HTTP connection with chunked Transfer-Encoding
        to server:port with optional headers.
        '''
        server = self._server
        port = self._port
        headers = self._headers
        self._conn = httplib.HTTPConnection(server, port)

        self._conn.putrequest('POST', '/')
        self._conn.putheader('Transfer-Encoding', 'chunked')
        for header in headers:
            self._conn.putheader(header, headers[header])
        self._conn.endheaders()

        # Set blocking to False prevents recv
        # from blocking while waiting for a response.
        self._conn.sock.setblocking(False)
        self._bytes = ''
        self._reset_retries()
        time.sleep(0.5)

    def close(self):
        ''' Close the connection to server.

        If available, return a httplib.HTTPResponse object.

        Closing the connection involves sending the
        Transfer-Encoding terminating bytes.
        '''
        self._reset_retries()
        self._closed = True

        # Chunked-encoded posts are terminated with '0\r\n\r\n'
        # For some reason, either Python or node.js seems to
        # require an extra \r\n.
        try:
            self._conn.send('\r\n0\r\n\r\n')
        except httplib.socket.error:
            # In case the socket has already been closed
            return ''

        return self._getresponse()

    def _getresponse(self):
        ''' Read from recv and return a HTTPResponse object if possible.
        Either
        1 - The client has succesfully closed the connection: Return ''
        2 - The server has already closed the connection: Return the response
            if possible.
        '''
        # Wait for a response
        self._conn.sock.setblocking(True)
        # Parse the response
        response = self._bytes
        while True:
            try:
                bytes = self._conn.sock.recv(1)
            except httplib.socket.error:
                # For error 54: Connection reset by peer
                # (and perhaps others)
                return ''
            if bytes == '':
                break
            else:
                response += bytes
        # Set recv to be non-blocking again
        self._conn.sock.setblocking(False)

        # Convert the response string to a httplib.HTTPResponse
        # object with a bit of a hack
        if response != '':
            # Taken from
            # http://pythonwise.blogspot.ca/2010/02/parse-http-response.html
            try:
                response = httplib.HTTPResponse(_FakeSocket(response))
                response.begin()
            except:
                # Bad headers ... etc.
                response = ''
        return response

    def _isconnected(self):
        ''' Return True if the socket is still connected
        to the server, False otherwise.

        This check is done in 3 steps:
        1 - Check if we have closed the connection
        2 - Check if the original socket connection failed
        3 - Check if the server has returned any data. If they have,
            assume that the server closed the response after they sent
            the data, i.e. that the data was the HTTP response.
        '''

        # 1 - check if we've closed the connection.
        if self._closed:
            return False

        # 2 - Check if the original socket connection failed
        # If this failed, then no socket was initialized
        if self._conn.sock is None:
            return False

        try:
            # 3 - Check if the server has returned any data.
            # If they have, then start to store the response
            # in _bytes.
            self._bytes = ''
            self._bytes = self._conn.sock.recv(1)
            return False
        except httplib.socket.error as e:
            # Check why recv failed
            if e.errno == 35:
                # This is the "Resource temporarily unavailable" error
                # which is thrown cuz there was nothing to receive, i.e.
                # the server hasn't returned a response yet.
                # So, assume that the connection is still open.
                return True
            elif e.errno == 54:
                # This is the "Connection reset by peer" error
                # which is thrown cuz the server reset the
                # socket, so the connection is closed.
                return False
            else:
                # Unknown scenario
                raise e

    def _reconnect(self):
        ''' Connect if disconnected.
        Retry self.maxtries times with delays
        '''
        if not self._isconnected():
            try:
                self._connect()
            except httplib.socket.error as e:
                # Attempt to reconnect if the connection was refused
                if e.errno == 61:
                    # errno 61 is the "Connection Refused" error
                    time.sleep(self._delay)
                    self._delay += self._delay  # fibonacii delays
                    self._tries += 1
                    if self._tries < self.maxtries:
                        self._reconnect()
                    else:
                        self._reset_retries()
                        raise e
                else:
                    # Unknown scenario
                    raise e

        # Reconnect worked - reset _closed
        self._closed = False

    def _reset_retries(self):
        ''' Reset the connect counters and delays
        '''
        self._tries = 0
        self._delay = 1


class _FakeSocket(StringIO.StringIO):
    # Used to construct a httplib.HTTPResponse object
    # from a string.
    # Thx to: http://pythonwise.blogspot.ca/2010/02/parse-http-response.html
    def makefile(self, *args, **kwargs):
        return self
