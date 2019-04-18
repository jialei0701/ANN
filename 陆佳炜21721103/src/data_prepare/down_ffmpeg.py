import imageio
import ssl

# 下面这一句不是必须的, 但是某些情况下访问 https 会报SSL证书不受信任, 加上这一句可以允许通过
ssl._create_default_https_context = ssl._create_unverified_context

# 下载 ffmpeg 组件
imageio.plugins.ffmpeg.download()