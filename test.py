import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import savgol_filter
from fastdtw import fastdtw
from filterpy.kalman import KalmanFilter
import warnings
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import json
import os
import tempfile
import logging

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 忽略protobuf兼容性警告
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# ----------------------
# 定义人体关节结构
# ----------------------
class HumanJoints:
    # MediaPipe的关节索引定义
    JOINTS = {
        # 躯干
        'nose': 0,
        'neck': 12,  # 近似值，实际是左右肩的中点
        'spine_mid': 24,  # 近似值，实际是左右髋的中点
        
        # 左侧关节
        'left_shoulder': 11,
        'left_elbow': 13,
        'left_wrist': 15,
        'left_hip': 23,
        'left_knee': 25,
        'left_ankle': 27,
        
        # 右侧关节
        'right_shoulder': 12,
        'right_elbow': 14,
        'right_wrist': 16,
        'right_hip': 24,
        'right_knee': 26,
        'right_ankle': 28,
    }
    
    # 定义关节链
    CHAINS = {
        # 上肢链
        'left_arm': ['left_shoulder', 'left_elbow', 'left_wrist'],
        'right_arm': ['right_shoulder', 'right_elbow', 'right_wrist'],
        
        # 下肢链
        'left_leg': ['left_hip', 'left_knee', 'left_ankle'],
        'right_leg': ['right_hip', 'right_knee', 'right_ankle'],
        
        # 躯干链
        'spine': ['nose', 'neck', 'spine_mid'],
        
        # 肩部链
        'shoulders': ['left_shoulder', 'right_shoulder'],
        
        # 髋部链
        'hips': ['left_hip', 'right_hip']
    }
    
    # 关节评估权重
    WEIGHTS = {
        'shoulders': 0.15,  # 肩部稳定性
        'hips': 0.15,      # 髋部稳定性
        'spine': 0.2,      # 脊柱姿态
        'left_arm': 0.125,  # 左臂动作
        'right_arm': 0.125, # 右臂动作
        'left_leg': 0.125,  # 左腿动作
        'right_leg': 0.125  # 右腿动作
    }
    
    @staticmethod
    def get_joint_indices():
        """获取所有需要跟踪的关节索引"""
        return list(set(HumanJoints.JOINTS.values()))
    
    @staticmethod
    def get_chain_indices(chain_name):
        """获取指定链的关节索引列表"""
        return [HumanJoints.JOINTS[joint] for joint in HumanJoints.CHAINS[chain_name]]

# ----------------------
# 模块1：视频解析与关节点提取
# ----------------------
class PoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        self.kalman_filters = {i: self._create_kalman() for i in HumanJoints.get_joint_indices()}

    def _create_kalman(self):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]])
        kf.H = np.array([[1,0,0,0], [0,1,0,0]])
        kf.P *= 1000
        kf.R = 5
        return kf

    def _preprocess_frame(self, frame):
        """预处理帧"""
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    def process_video(self, video_path):
        """处理视频并提取姿势数据"""
        logger.info(f"开始处理视频: {video_path}")
        cap = cv2.VideoCapture(video_path)
        poses = []
        prev_valid = {}
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                if frame is None or frame.size == 0:
                    continue
                
                frame = self._preprocess_frame(frame)
                results = self.pose.process(frame)
                landmarks = results.pose_landmarks
                
                current_pose = {}
                if landmarks:
                    for idx in HumanJoints.get_joint_indices():
                        lm = landmarks.landmark[idx]
                        if lm.visibility < 0.1:
                            if idx in prev_valid:
                                x, y = prev_valid[idx]
                            else:
                                continue
                        else:
                            self.kalman_filters[idx].predict()
                            self.kalman_filters[idx].update([[lm.x], [lm.y]])
                            x, y = self.kalman_filters[idx].x[:2].flatten()
                            prev_valid[idx] = (x, y)
                        
                        current_pose[idx] = [float(x), float(y), float(lm.visibility)]
                
                poses.append(current_pose)
        except Exception as e:
            logger.error(f"处理视频时出错: {str(e)}")
            raise
        finally:
            cap.release()
            
        logger.info(f"视频处理完成，共 {len(poses)} 帧")
        return poses

# ----------------------
# 模块2：动作评估器
# ----------------------
class MotionEvaluator:
    def __init__(self):
        self.chains = HumanJoints.CHAINS
        self.weights = HumanJoints.WEIGHTS

    def _calc_chain_metrics(self, points1, points2):
        """计算关节链的多个指标"""
        if len(points1) != len(points2) or len(points1) < 2:
            return None
            
        metrics = {
            'angle_diff': 0,    # 关节角度差异
            'length_ratio': 0,  # 关节链长度比
            'position_diff': 0  # 位置差异
        }
        
        # 计算角度差异
        if len(points1) >= 3:
            angle1 = self._calc_angle(points1[0], points1[1], points1[2])
            angle2 = self._calc_angle(points2[0], points2[1], points2[2])
            metrics['angle_diff'] = abs(angle1 - angle2)
        
        # 计算长度比
        len1 = self._calc_chain_length(points1)
        len2 = self._calc_chain_length(points2)
        metrics['length_ratio'] = abs(len1/len2 - 1) if len2 > 0 else 1
        
        # 计算位置差异
        metrics['position_diff'] = np.mean([
            np.linalg.norm(np.array(p1) - np.array(p2))
            for p1, p2 in zip(points1, points2)
        ])
        
        return metrics

    def _calc_angle(self, p1, p2, p3):
        """计算三点角度"""
        if None in [p1, p2, p3]:
            return 0
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    def _calc_chain_length(self, points):
        """计算关节链总长度"""
        return sum(
            np.linalg.norm(np.array(points[i]) - np.array(points[i-1]))
            for i in range(1, len(points))
        )

    def analyze(self, user_poses, ref_poses, alignment):
        """分析用户动作与参考动作的差异"""
        logger.info("开始动作分析")
        chain_scores = {chain: [] for chain in self.chains.keys()}
        
        for u_idx, r_idx in zip(alignment[0], alignment[1]):
            for chain_name, joint_names in self.chains.items():
                indices = HumanJoints.get_chain_indices(chain_name)
                user_points = [user_poses[u_idx].get(idx, (0,0,0))[:2] for idx in indices]
                ref_points = [ref_poses[r_idx].get(idx, (0,0,0))[:2] for idx in indices]
                
                metrics = self._calc_chain_metrics(user_points, ref_points)
                if metrics:
                    chain_score = 100 - (
                        metrics['angle_diff'] * 0.4 +
                        metrics['length_ratio'] * 100 * 0.3 +
                        metrics['position_diff'] * 100 * 0.3
                    )
                    chain_scores[chain_name].append(max(0, min(100, chain_score)))

        final_metrics = {}
        total_score = 0
        
        for chain_name, scores in chain_scores.items():
            if scores:
                chain_avg = np.mean(scores)
                final_metrics[chain_name] = {
                    'score': float(chain_avg),
                    'stability': float(np.std(scores)),
                }
                total_score += chain_avg * self.weights[chain_name]

        logger.info("动作分析完成")
        return {
            'total_score': float(max(0, min(100, total_score))),
            'chain_metrics': final_metrics,
            'time_sync': float(len(alignment[0]) / max(len(ref_poses), 1))
        }

# ----------------------
# 模块3：DTW对齐器
# ----------------------
class DTWAligner:
    def __init__(self):
        pass
        
    def align(self, user_poses, ref_poses):
        """使用DTW算法对齐两个姿势序列"""
        logger.info("开始序列对齐")
        user_features = self._poses_to_features(user_poses)
        ref_features = self._poses_to_features(ref_poses)
        
        _, path = fastdtw(user_features, ref_features)
        logger.info("序列对齐完成")
        return list(zip(*path))
    
    def _poses_to_features(self, poses):
        """将姿势转换为特征向量"""
        features = []
        for pose in poses:
            feature = []
            for idx in HumanJoints.get_joint_indices():
                if idx in pose:
                    feature.extend(pose[idx][:2])
                else:
                    feature.extend([0, 0])
            features.append(feature)
        return np.array(features)

# ----------------------
# Flask应用
# ----------------------
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# 配置上传文件夹
UPLOAD_FOLDER = tempfile.mkdtemp()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB

def allowed_file(filename):
    """检查文件类型是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov'}

@app.route('/')
def index():
    """渲染主页"""
    return render_template('kinetic_chain.html')

@app.route('/api/poses', methods=['POST'])
def analyze_poses():
    """处理视频上传和分析请求"""
    logger.info("收到新的分析请求")
    
    if 'user_video' not in request.files or 'ref_video' not in request.files:
        logger.error("未找到上传的视频文件")
        return jsonify({'error': '请上传两个视频文件'}), 400
    
    user_video = request.files['user_video']
    ref_video = request.files['ref_video']
    
    if not user_video or not ref_video:
        logger.error("文件未选择")
        return jsonify({'error': '文件未选择'}), 400
    
    if not allowed_file(user_video.filename) or not allowed_file(ref_video.filename):
        logger.error(f"不支持的文件格式: {user_video.filename}, {ref_video.filename}")
        return jsonify({'error': '不支持的文件格式'}), 400

    try:
        # 保存上传的视频文件
        user_path = os.path.join(app.config['UPLOAD_FOLDER'], 'user_temp.mp4')
        ref_path = os.path.join(app.config['UPLOAD_FOLDER'], 'ref_temp.mp4')
        
        logger.info(f"保存视频文件到: {user_path}, {ref_path}")
        user_video.save(user_path)
        ref_video.save(ref_path)
        
        # 处理视频
        analyzer = PoseAnalyzer()
        user_poses = analyzer.process_video(user_path)
        ref_poses = analyzer.process_video(ref_path)
        
        # 对齐序列
        aligner = DTWAligner()
        alignment = aligner.align(user_poses, ref_poses)
        
        # 评估动作
        evaluator = MotionEvaluator()
        report = evaluator.analyze(user_poses, ref_poses, alignment)
        
        logger.info("分析完成，准备返回结果")
        return jsonify({
            'user_poses': user_poses,
            'ref_poses': ref_poses,
            'report': report
        })

    except Exception as e:
        logger.exception("处理过程中出现错误")
        return jsonify({'error': str(e)}), 500
        
    finally:
        # 清理临时文件
        logger.info("清理临时文件")
        if os.path.exists(user_path):
            os.remove(user_path)
        if os.path.exists(ref_path):
            os.remove(ref_path)

# 错误处理
@app.errorhandler(413)
def too_large(e):
    """处理文件过大的错误"""
    return jsonify({'error': '文件大小超过限制'}), 413

@app.errorhandler(500)
def internal_error(e):
    """处理服务器内部错误"""
    return jsonify({'error': '服务器内部错误'}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
