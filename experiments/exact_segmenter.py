import numpy as np
import itertools as it

class Segmenter:

	def __init__(self, V):

		# vocabulary
		self.V = V

	def segment(self, input):
		# semi-Markov dynamic program for shortest segmentation with max frequency

		N = len(input)

		# dynamic-programming array
		𝜷 = np.ones((N+1, 2), dtype=int)*N 
		𝜷[0] = 0
		𝜷[:, 1] = 0

		# backpointers
		bp = { 0 : -1}

		for m in range(N+1):
			for n in range(m):
				segment = input[n:m]
				if segment in self.V: 
					if 𝜷[n, 0] + 1 < 𝜷[m, 0]:
						𝜷[m, 0] = 𝜷[n, 0] + 1
						𝜷[m, 1] = 𝜷[n, 1] + V[segment]
						bp[m] = n
					elif 𝜷[m, 0] ==  𝜷[n, 0] + 1:
						𝜷[m, 1] = max(𝜷[m, 1], 𝜷[n, 1] + V[segment])
						# TODO: this backpointer is a bit off; see Amit's version
						bp[m] = n

		# extract backpointers
		ptr = N
		segmentation =[]
		while ptr != 0:
			segmentation = [input[bp[ptr]:ptr]] + segmentation
			ptr = bp[ptr]

		# make sure we didn't fuck it
		assert 𝜷[N, 1] == sum(map(lambda x: self.V[x], segmentation))
		assert "".join(segmentation) == input

		return segmentation, 𝜷[N, 0], 𝜷[N, 1]

	def brute(self, input):
		"""	 
		Brute-force method to compute the shortest segmentation
		such that every segment is in the vocabulary
		"""
		N = len(input)
		min_len, max_freq = N, 0

		for mask in it.product([0, 1], repeat=N):
			segmentation = []
			cur = ""
			for i, c in zip(mask, input):
				if i == 1:
					segmentation.append(cur)
					cur = c
				else:
					cur += c
			if cur != "":
				segmentation.append(cur)

			# check whether each segment is in the vocabulary
			bad = False
			for segment in segmentation:
				if segment not in self.V:
					bad = True
			if not bad:
				freq = sum(map(lambda x: self.V[x], segmentation))
				if len(segmentation) < min_len:
					min_len = len(segmentation)
					max_freq = freq
				elif len(segmentation) == min_len:
					max_freq = max(max_freq, freq)

			assert "".join(segmentation) == input, print(segmentation, mask)

		return min_len, max_freq

V = { "a" : 5, "b" : 9, "c" : 50, "ab" : 11, "bc" : 20, "cb" : 31 }

segmenter = Segmenter(V)
print(segmenter.segment("abccbaba"))
print(segmenter.brute("abccbaba"))
