import numpy as np
import torch

class DRU:
	# sigma: độ lệch chuẩn của nhiễu Gaussian thêm vào messeage 
	# comm_narrow : chế độ kênh hẹp. Nếu True thì message là bit nhị phân -> sigmoid, nếu False thì 
	# message là vector đa lớp và dùng softmax.
	# hard: có dùng chế độ cứng khi discretize hay không 
	# Nếu True -> threshold , nếu False -> Sharpened Sigmoid 

	# Tín hiệu liên tục (hard = False ) được sử dụng trong quá trình learning
	# Đinh nghĩa các dữ liệu hơi cứng: tensor([1.9287e-09, 1.0000e+00, 1.0000e+00, 1.0000e+00])
	def __init__(self, sigma, comm_narrow=True, hard=False):
		self.sigma = sigma
		self.comm_narrow = comm_narrow
		self.hard = hard

	def regularize(self, m):
		# Chuyển m thành một phân phối N(m, sigma ^ 2)	
		m_reg = m + torch.randn(m.size()) * self.sigma
		if self.comm_narrow:
			m_reg = torch.sigmoid(m_reg)
		else:
			m_reg = torch.softmax(m_reg, 0)
		return m_reg

	def discretize(self, m):
		if self.hard:
			if self.comm_narrow:
				# gt: greater than : so sánh với 0.5 để tạo ra một mặ nạ : True False
				# float (): chuyển True , False thành 1.0  và 0.0 
				# sign: Lấy dấu của con số 

				return (m.gt(0.5).float() - 0.5).sign().float()
			else:
				# Tạo ra một bản sao rỗng 
				m_ = torch.zeros_like(m)
				# Thực hiện One-Hot để chuyển phân phối liên tục thành phân phối rời rạc
				# ví dụ: [0.5, 0.3, 0.1] -> [1,0,0]
				if m.dim() == 1:      
					_, idx = m.max(0)
					m_[idx] = 1.
				elif m.dim() == 2:      
					_, idx = m.max(1)
					for b in range(idx.size(0)):
						m_[b, idx[b]] = 1.
				else:
					raise ValueError('Wrong message shape: {}'.format(m.size()))
				return m_
		else:
			scale = 2 * 20
			if self.comm_narrow:
				# Biến scale biến hàm sigmoid hay softmax trở nên rất dốc biến các giá trị gần 0.5 về hẳn 1 hoặc 0
				return torch.sigmoid((m.gt(0.5).float() - 0.5) * scale)
			else:
				return torch.softmax(m * scale, -1)

# Nếu train_mode = True thì thực hiện chuẩn hóa về dạng vector liên tục để có thể truyền gradient về
# Lúc thực thi thì chuyển tiếp các tín hiệu rời rạc để tiết kiệm băng thông
	def forward(self, m, train_mode):
		if train_mode:
			return self.regularize(m)
		else:
			return self.discretize(m)
			
# Việc tạo ra hai kênh giúp cho việc tổi thiệu băng thông , có thể lựa chọn kênh truyền 
