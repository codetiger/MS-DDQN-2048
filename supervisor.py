import ctypes
import time
import os
from multiprocessing.pool import ThreadPool

class Supervisor:
	def __init__(self, gridSize):
		dllfn = 'bin/2048.so'
		self.ailib = ctypes.CDLL(dllfn)
		self.ailib.init_tables()

		self.ailib.find_best_move.argtypes = [ctypes.c_uint64]
		self.ailib.score_toplevel_move.argtypes = [ctypes.c_uint64, ctypes.c_int]
		self.ailib.score_toplevel_move.restype = ctypes.c_float

		self.gridSize = gridSize
		self.pool = ThreadPool(4)

	def to_c_board(self, m):
		board = 0
		i = 0
		for row in m:
			for c in row:            
				board |= c << (4*i)
				i += 1
		return board

	def _to_val(self, c):
		if c == 0: return 0
		return 2**c

	def to_val(self, m):
		return [[self._to_val(c) for c in row] for row in m]

	def _to_score(self, c):
		if c <= 1:
			return 0
		return (c-1) * (2**c)

	def to_score(self, m):
		return [[self._to_score(c) for c in row] for row in m]

	def find_best_move(self, m):
		board = self.to_c_board(m)
		#[0:'Right', 1:'Up', 2:'Down', 3:'Left']
		#['up', 'down', 'left', 'right']
		# dt = {0:2, 1:3, 2:0, 3:1}
		dt = {1:0, 3:1, 0:2, 2:3}
		move = self.ailib.find_best_move(board)
		return dt[move]

	def score_toplevel_move(self, args):
		return self.ailib.score_toplevel_move(*args)

	def find_best_move_mt(self, m):
		board = self.to_c_board(m)

		scores = self.pool.map(self.score_toplevel_move, [(board, move) for move in range(4)])

		bestmove, bestscore = max(enumerate(scores), key=lambda x:x[1])
		if bestscore == 0:
			return -1

		dt = {1:0, 3:1, 0:2, 2:3}
		return dt[bestmove]
