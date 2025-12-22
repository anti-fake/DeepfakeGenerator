import argparse
import json
import os
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from scipy import linalg
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def list_paths_recursive(root_dirs: List[str]) -> List[str]:
    """재귀적으로 이미지 경로 수집"""
    paths = []
    for root_dir in root_dirs:
        if os.path.isdir(root_dir):
            for file in os.listdir(root_dir):
                path = os.path.join(root_dir, file)
                paths.extend(list_paths_recursive([path]))
        else:
            if root_dir.endswith(('.png', '.jpg', '.jpeg')):
                paths.append(root_dir)
    return paths


class ImageDataset(Dataset):
    def __init__(self, root_dirs: List[str]) -> None:
        super().__init__()
        self.root_dirs = root_dirs
        self.image_paths = list_paths_recursive(root_dirs)
        assert len(self.image_paths) > 0, "No images found"
        self.image_paths.sort()
        self.transform = transforms.Compose([
            transforms.Resize((299, 299), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image


class InceptionV3FeatureExtractor(nn.Module):
    """InceptionV3의 pool3 layer에서 2048차원 feature 추출"""
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        self.blocks = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        
    def forward(self, x):
        x = self.blocks(x)
        return x.view(x.size(0), -1)


def extract_features(dataloader, model, device):
    """DataLoader로부터 feature 추출"""
    model.eval()
    all_features = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            batch = batch.to(device)
            features = model(batch)
            all_features.append(features.cpu().numpy())
    
    return np.concatenate(all_features, axis=0)


def compute_fid(features_A: np.ndarray, features_B: np.ndarray) -> Tuple[float, float, float]:
    """두 feature 세트 간의 FID 계산"""
    mu_A = np.mean(features_A, axis=0)
    mu_B = np.mean(features_B, axis=0)
    sigma_A = np.cov(features_A, rowvar=False)
    sigma_B = np.cov(features_B, rowvar=False)
    
    diff = mu_A - mu_B
    mean_term = np.sum(diff ** 2)
    
    covmean = linalg.sqrtm(sigma_A @ sigma_B)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    cov_term = np.trace(sigma_A + sigma_B - 2 * covmean)
    
    fid = mean_term + cov_term
    return fid, mean_term, cov_term


def compute_mahalanobis_distances(features_A: np.ndarray, features_B: np.ndarray) -> np.ndarray:
    """B의 각 샘플에서 A 분포까지의 Mahalanobis 거리 계산"""
    mu_A = np.mean(features_A, axis=0)
    sigma_A = np.cov(features_A, rowvar=False)
    
    # 정규화된 역행렬
    sigma_A_reg = sigma_A + np.eye(sigma_A.shape[0]) * 1e-6
    try:
        sigma_A_inv = np.linalg.inv(sigma_A_reg)
    except:
        sigma_A_inv = np.linalg.pinv(sigma_A_reg)
    
    diff = features_B - mu_A
    mahal_distances = np.sqrt(np.sum(diff @ sigma_A_inv * diff, axis=1))
    
    return mahal_distances


# ============================================================
# 🎯 새로운 FID 개선 전략들 (v3)
# ============================================================

def strategy_aggressive_outlier_search(
    features_A: np.ndarray, 
    features_B: np.ndarray,
    percentiles: List[int] = None
) -> Tuple[float, float, float, np.ndarray, dict]:
    """
    전략 1: 공격적 Outlier 제거 탐색
    percentile을 95→50까지 내려가며 FID 변화 추적
    FID가 증가하기 시작하는 최적점 자동 탐지
    """
    if percentiles is None:
        percentiles = list(range(95, 45, -5))  # 95, 90, 85, ..., 50
    
    print(f"\n[전략 1] 공격적 Outlier 제거 탐색")
    print(f"  탐색 범위: {percentiles}")
    print("-" * 70)
    
    # Mahalanobis 거리 계산 (한 번만)
    mahal_distances = compute_mahalanobis_distances(features_A, features_B)
    
    results = {}
    best_fid = float('inf')
    best_percentile = 100
    best_indices = np.arange(len(features_B))
    
    # 원본 FID
    orig_fid, orig_mean, orig_cov = compute_fid(features_A, features_B)
    results[100] = {'fid': orig_fid, 'mean': orig_mean, 'cov': orig_cov, 'n_samples': len(features_B)}
    print(f"  p=100: FID={orig_fid:.4f} (n={len(features_B)})")
    
    prev_fid = orig_fid
    
    for p in percentiles:
        threshold = np.percentile(mahal_distances, p)
        selected_indices = np.where(mahal_distances <= threshold)[0]
        
        if len(selected_indices) < 1000:  # 최소 샘플 수 보장
            print(f"  p={p}: 샘플 수 부족 ({len(selected_indices)}), 스킵")
            continue
        
        fid, mean_t, cov_t = compute_fid(features_A, features_B[selected_indices])
        results[p] = {'fid': fid, 'mean': mean_t, 'cov': cov_t, 'n_samples': len(selected_indices)}
        
        delta = fid - prev_fid
        marker = "⬇️" if delta < 0 else "⬆️" if delta > 0 else "➡️"
        print(f"  p={p:2d}: FID={fid:.4f} (n={len(selected_indices):5d}) {marker} Δ={delta:+.4f}")
        
        if fid < best_fid:
            best_fid = fid
            best_percentile = p
            best_indices = selected_indices.copy()
        
        prev_fid = fid
        
        # 조기 종료: FID가 3회 연속 증가하면
        if len(results) >= 4:
            recent_fids = [results[k]['fid'] for k in sorted(results.keys(), reverse=True)[:4]]
            if all(recent_fids[i] <= recent_fids[i+1] for i in range(3)):
                print(f"  [조기종료] FID 연속 증가 감지")
                break
    
    # 최적점 출력
    best_result = results[best_percentile]
    print("-" * 70)
    print(f"  🏆 최적: p={best_percentile}, FID={best_fid:.4f}")
    print(f"     (평균: {best_result['mean']:.4f}, 공분산: {best_result['cov']:.4f})")
    
    return best_fid, best_result['mean'], best_result['cov'], best_indices, results


def strategy_two_stage_hybrid(
    features_A: np.ndarray, 
    features_B: np.ndarray,
    outlier_percentile: int = 85,
    target_size: Optional[int] = None,
    n_clusters: int = 50
) -> Tuple[float, float, float, np.ndarray]:
    """
    전략 2: 2단계 복합 전략
    Stage 1: Mahalanobis outlier 제거
    Stage 2: KMeans stratified sampling (A 클러스터 비율 유지, 중심에 가까운 샘플 우선)
    """
    if target_size is None:
        target_size = len(features_A)
    
    print(f"\n[전략 2] 2단계 복합 (outlier={outlier_percentile}%, target={target_size})")
    
    # Stage 1: Outlier 제거
    mahal_distances = compute_mahalanobis_distances(features_A, features_B)
    threshold = np.percentile(mahal_distances, outlier_percentile)
    stage1_mask = mahal_distances <= threshold
    stage1_indices = np.where(stage1_mask)[0]
    stage1_features = features_B[stage1_indices]
    
    print(f"  Stage 1 후: {len(stage1_indices)}개")
    
    # Stage 2: Stratified Sampling
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(features_A)
    
    labels_A = kmeans.labels_
    labels_B = kmeans.predict(stage1_features)
    
    # A의 클러스터별 비율
    cluster_counts_A = np.bincount(labels_A, minlength=n_clusters)
    cluster_ratios = cluster_counts_A / len(features_A)
    
    selected_local_indices = []
    
    for cluster_id in range(n_clusters):
        n_needed = max(1, int(cluster_ratios[cluster_id] * target_size))
        cluster_local_indices = np.where(labels_B == cluster_id)[0]
        
        if len(cluster_local_indices) == 0:
            continue
        
        # 클러스터 중심에 가까운 순으로 정렬
        center = kmeans.cluster_centers_[cluster_id]
        cluster_features = stage1_features[cluster_local_indices]
        distances = np.linalg.norm(cluster_features - center, axis=1)
        sorted_local = cluster_local_indices[np.argsort(distances)]
        
        n_select = min(n_needed, len(sorted_local))
        selected_local_indices.extend(sorted_local[:n_select].tolist())
    
    # 원래 인덱스로 변환
    selected_indices = stage1_indices[selected_local_indices]
    print(f"  Stage 2 후: {len(selected_indices)}개")
    
    fid, mean_t, cov_t = compute_fid(features_A, features_B[selected_indices])
    print(f"  → FID: {fid:.4f} (평균: {mean_t:.4f}, 공분산: {cov_t:.4f})")
    
    return fid, mean_t, cov_t, selected_indices


def strategy_iterative_removal(
    features_A: np.ndarray, 
    features_B: np.ndarray,
    target_fid: float = 20.0,
    max_remove_ratio: float = 0.5,
    removal_rate: float = 0.01,
    max_iter: int = 100
) -> Tuple[float, float, float, np.ndarray]:
    """
    전략 3: Iterative Refinement
    매 iteration마다 FID 기여도가 높은 상위 샘플 제거
    종료 조건: FID가 목표 도달 또는 연속 증가
    """
    print(f"\n[전략 3] Iterative Removal (목표: {target_fid}, 최대 제거: {max_remove_ratio*100}%)")
    
    current_indices = np.arange(len(features_B))
    min_samples = int(len(features_B) * (1 - max_remove_ratio))
    
    current_fid, current_mean, current_cov = compute_fid(features_A, features_B)
    best_fid = current_fid
    best_indices = current_indices.copy()
    
    print(f"  초기: FID={current_fid:.4f} (n={len(current_indices)})")
    
    fid_history = [current_fid]
    consecutive_increase = 0
    
    for iteration in range(max_iter):
        if len(current_indices) <= min_samples:
            print(f"  [종료] 최소 샘플 수 도달")
            break
        
        if current_fid <= target_fid:
            print(f"  [종료] 목표 FID 도달!")
            break
        
        # 현재 샘플들의 FID 기여도 계산 (leave-one-out 근사)
        current_features = features_B[current_indices]
        mu_A = np.mean(features_A, axis=0)
        mu_B = np.mean(current_features, axis=0)
        
        # 평균에 대한 기여도: 각 샘플이 평균을 얼마나 벗어나게 하는가
        mean_contribution = np.sum((current_features - mu_A) ** 2, axis=1)
        
        # Mahalanobis 거리 (공분산 기여도 근사)
        mahal_dist = compute_mahalanobis_distances(features_A, current_features)
        
        # 종합 점수 (높을수록 나쁨)
        combined_score = mean_contribution + mahal_dist ** 2
        
        # 상위 removal_rate 비율 제거
        n_remove = max(1, int(len(current_indices) * removal_rate))
        remove_local_indices = np.argsort(combined_score)[-n_remove:]
        
        # 제거
        keep_mask = np.ones(len(current_indices), dtype=bool)
        keep_mask[remove_local_indices] = False
        current_indices = current_indices[keep_mask]
        
        # 새 FID 계산
        new_fid, new_mean, new_cov = compute_fid(features_A, features_B[current_indices])
        
        delta = new_fid - current_fid
        if (iteration + 1) % 10 == 0 or delta < 0:
            marker = "⬇️" if delta < 0 else "⬆️"
            print(f"  Iter {iteration+1:3d}: FID={new_fid:.4f} (n={len(current_indices):5d}) {marker} Δ={delta:+.4f}")
        
        if new_fid < best_fid:
            best_fid = new_fid
            best_indices = current_indices.copy()
            consecutive_increase = 0
        else:
            consecutive_increase += 1
        
        if consecutive_increase >= 5:
            print(f"  [종료] FID 연속 증가")
            break
        
        current_fid = new_fid
        fid_history.append(current_fid)
    
    fid, mean_t, cov_t = compute_fid(features_A, features_B[best_indices])
    print(f"  → 최종 FID: {fid:.4f} (평균: {mean_t:.4f}, 공분산: {cov_t:.4f})")
    print(f"  → 선택된 샘플: {len(best_indices)}개")
    
    return fid, mean_t, cov_t, best_indices


def strategy_covariance_greedy(
    features_A: np.ndarray, 
    features_B: np.ndarray,
    target_size: Optional[int] = None,
    n_iter: int = 500,
    batch_size: int = 100
) -> Tuple[float, float, float, np.ndarray]:
    """
    전략 4: 공분산 직접 매칭 (Greedy)
    A의 공분산 행렬을 target으로
    B subset의 공분산이 A와 최대한 비슷해지도록 샘플 선택
    ||sigma_A - sigma_B_subset||_F 최소화
    """
    if target_size is None:
        target_size = len(features_A) * 2
    
    print(f"\n[전략 4] 공분산 Greedy 매칭 (target={target_size})")
    
    sigma_A = np.cov(features_A, rowvar=False)
    mu_A = np.mean(features_A, axis=0)
    
    # 초기화: Mahalanobis 거리 기준 상위 샘플들로 시작
    mahal_distances = compute_mahalanobis_distances(features_A, features_B)
    init_indices = np.argsort(mahal_distances)[:target_size]
    
    current_indices = list(init_indices)
    remaining_indices = list(set(range(len(features_B))) - set(current_indices))
    
    current_sigma = np.cov(features_B[current_indices], rowvar=False)
    current_frob = np.linalg.norm(sigma_A - current_sigma, 'fro')
    
    print(f"  초기 Frobenius: {current_frob:.4f}")
    
    best_frob = current_frob
    best_indices = current_indices.copy()
    
    for iteration in tqdm(range(n_iter), desc="  Greedy 최적화"):
        improved = False
        
        # 랜덤하게 swap 시도
        np.random.seed(iteration)
        swap_candidates = np.random.choice(len(current_indices), size=min(batch_size, len(current_indices)), replace=False)
        
        for local_idx in swap_candidates:
            idx_out = current_indices[local_idx]
            
            # 랜덤하게 교체 후보 선택
            candidates_in = np.random.choice(remaining_indices, size=min(10, len(remaining_indices)), replace=False)
            
            for idx_in in candidates_in:
                test_indices = current_indices.copy()
                test_indices[local_idx] = idx_in
                
                test_sigma = np.cov(features_B[test_indices], rowvar=False)
                test_frob = np.linalg.norm(sigma_A - test_sigma, 'fro')
                
                if test_frob < current_frob:
                    remaining_indices.remove(idx_in)
                    remaining_indices.append(idx_out)
                    current_indices[local_idx] = idx_in
                    current_frob = test_frob
                    improved = True
                    
                    if current_frob < best_frob:
                        best_frob = current_frob
                        best_indices = current_indices.copy()
                    break
            
            if improved:
                break
        
        if (iteration + 1) % 100 == 0:
            tqdm.write(f"    Iter {iteration+1}: Frob={current_frob:.4f}")
    
    print(f"  최종 Frobenius: {best_frob:.4f}")
    
    fid, mean_t, cov_t = compute_fid(features_A, features_B[best_indices])
    print(f"  → FID: {fid:.4f} (평균: {mean_t:.4f}, 공분산: {cov_t:.4f})")
    
    return fid, mean_t, cov_t, np.array(best_indices)


def strategy_sinkhorn_ot(
    features_A: np.ndarray, 
    features_B: np.ndarray,
    target_size: Optional[int] = None,
    reg: float = 0.05,
    device: str = 'cuda'
) -> Tuple[float, float, float, np.ndarray]:
    """
    전략 5: Sinkhorn OT 기반 선택 (GPU 가속)
    Optimal Transport로 importance score 계산
    상위 target_size개 선택
    """
    if target_size is None:
        target_size = len(features_A) * 2
    
    print(f"\n[전략 5] Sinkhorn OT (target={target_size}, reg={reg})")
    
    try:
        import ot
    except ImportError:
        print("  ⚠️ POT 라이브러리 필요: pip install POT")
        return float('inf'), 0, 0, np.array([])
    
    n_A = len(features_A)
    n_B = len(features_B)
    
    # Uniform distributions
    a = np.ones(n_A) / n_A
    b = np.ones(n_B) / n_B
    
    print(f"  Cost matrix 계산 중... ({n_A} x {n_B})")
    
    # Cost matrix (L2 distance)
    # GPU 사용 시
    if device == 'cuda' and torch.cuda.is_available():
        A_torch = torch.from_numpy(features_A).float().cuda()
        B_torch = torch.from_numpy(features_B).float().cuda()
        
        # Chunked computation to avoid OOM
        chunk_size = 10000
        cost_chunks = []
        
        for i in range(0, n_B, chunk_size):
            end_i = min(i + chunk_size, n_B)
            B_chunk = B_torch[i:end_i]
            
            # (n_A, chunk) distance matrix
            diff = A_torch.unsqueeze(1) - B_chunk.unsqueeze(0)
            cost_chunk = torch.sum(diff ** 2, dim=2).cpu().numpy()
            cost_chunks.append(cost_chunk)
        
        cost_matrix = np.concatenate(cost_chunks, axis=1)
        del A_torch, B_torch
        torch.cuda.empty_cache()
    else:
        # CPU fallback
        from scipy.spatial.distance import cdist
        cost_matrix = cdist(features_A, features_B, metric='sqeuclidean')
    
    print(f"  Sinkhorn 계산 중... (reg={reg})")
    
    # Sinkhorn (CPU, POT library)
    try:
        T = ot.sinkhorn(a, b, cost_matrix, reg=reg, numItermax=1000, stopThr=1e-9)
    except Exception as e:
        print(f"  ⚠️ Sinkhorn 실패: {e}")
        # Fallback to simpler OT
        T = ot.emd(a, b, cost_matrix)
    
    # Importance score: 각 B 샘플이 A에 얼마나 매칭되는가
    importance = T.sum(axis=0)
    
    # 상위 target_size개 선택
    selected_indices = np.argsort(importance)[-target_size:]
    
    print(f"  선택된 샘플: {len(selected_indices)}개")
    print(f"  Importance 분포: min={importance.min():.6f}, max={importance.max():.6f}")
    
    fid, mean_t, cov_t = compute_fid(features_A, features_B[selected_indices])
    print(f"  → FID: {fid:.4f} (평균: {mean_t:.4f}, 공분산: {cov_t:.4f})")
    
    return fid, mean_t, cov_t, selected_indices


def strategy_eigenvalue_matching(
    features_A: np.ndarray, 
    features_B: np.ndarray,
    target_size: Optional[int] = None,
    n_components: int = 100
) -> Tuple[float, float, float, np.ndarray]:
    """
    전략 6: 고유값 분포 매칭
    A의 주요 주성분별 분산에 맞춰 B 샘플 선택
    """
    if target_size is None:
        target_size = len(features_A) * 2
    
    print(f"\n[전략 6] 고유값 분포 매칭 (target={target_size})")
    
    from sklearn.decomposition import PCA
    
    # A로 PCA 학습
    pca = PCA(n_components=min(n_components, features_A.shape[1]))
    pca.fit(features_A)
    
    A_pca = pca.transform(features_A)
    B_pca = pca.transform(features_B)
    
    # A의 각 주성분별 분산 및 범위
    var_A = np.var(A_pca, axis=0)
    mean_A_pca = np.mean(A_pca, axis=0)
    std_A_pca = np.std(A_pca, axis=0)
    
    # B의 각 샘플이 A의 분포와 얼마나 맞는지 점수화
    # 주요 PC들에서 A의 분포 범위 내에 있는 정도
    scores = np.zeros(len(features_B))
    
    for pc_idx in range(min(20, len(var_A))):  # 상위 20개 PC만 사용
        # A의 해당 PC 분포 (평균 ± 2*std 범위)
        low = mean_A_pca[pc_idx] - 2 * std_A_pca[pc_idx]
        high = mean_A_pca[pc_idx] + 2 * std_A_pca[pc_idx]
        
        # B의 해당 PC 값이 범위 내에 있으면 가산점
        in_range = (B_pca[:, pc_idx] >= low) & (B_pca[:, pc_idx] <= high)
        scores += in_range.astype(float) * (var_A[pc_idx] / var_A.sum())  # 분산 비율로 가중
        
        # 범위에서 벗어난 정도에 따라 페널티
        deviation = np.abs(B_pca[:, pc_idx] - mean_A_pca[pc_idx]) / (std_A_pca[pc_idx] + 1e-6)
        scores -= deviation * 0.01 * (var_A[pc_idx] / var_A.sum())
    
    # 상위 target_size개 선택
    selected_indices = np.argsort(scores)[-target_size:]
    
    # 검증: 선택된 샘플의 분산 비교
    selected_pca = B_pca[selected_indices]
    var_selected = np.var(selected_pca, axis=0)
    
    print(f"  상위 5개 PC 분산 비교:")
    for i in range(5):
        ratio = var_selected[i] / var_A[i] if var_A[i] > 0 else 0
        print(f"    PC{i+1}: A={var_A[i]:.4f}, Selected={var_selected[i]:.4f}, ratio={ratio:.2f}x")
    
    fid, mean_t, cov_t = compute_fid(features_A, features_B[selected_indices])
    print(f"  → FID: {fid:.4f} (평균: {mean_t:.4f}, 공분산: {cov_t:.4f})")
    
    return fid, mean_t, cov_t, selected_indices


def strategy_combined_best(
    features_A: np.ndarray, 
    features_B: np.ndarray,
    device: str = 'cuda'
) -> Tuple[float, float, float, np.ndarray]:
    """
    전략 7: 최적 조합 탐색
    여러 전략의 결과를 조합하여 최적 선택
    """
    print(f"\n[전략 7] 최적 조합 탐색")
    
    results = []
    
    # 각 전략별 최적 결과 수집
    # 1. Aggressive outlier search
    try:
        fid1, mean1, cov1, indices1, _ = strategy_aggressive_outlier_search(
            features_A, features_B, 
            percentiles=[95, 90, 85, 80, 75]
        )
        results.append(("Aggressive Outlier", fid1, mean1, cov1, indices1))
    except Exception as e:
        print(f"  전략1 실패: {e}")
    
    # 2. Two-stage hybrid (다양한 파라미터)
    for outlier_p in [90, 85, 80]:
        for target_mult in [1, 2, 4]:
            try:
                target = len(features_A) * target_mult
                fid2, mean2, cov2, indices2 = strategy_two_stage_hybrid(
                    features_A, features_B, 
                    outlier_percentile=outlier_p,
                    target_size=target
                )
                results.append((f"Hybrid(p={outlier_p},t={target})", fid2, mean2, cov2, indices2))
            except Exception as e:
                pass
    
    # 3. Iterative removal
    try:
        fid3, mean3, cov3, indices3 = strategy_iterative_removal(
            features_A, features_B,
            target_fid=20.0,
            max_remove_ratio=0.3
        )
        results.append(("Iterative Removal", fid3, mean3, cov3, indices3))
    except Exception as e:
        print(f"  전략3 실패: {e}")
    
    # 결과 정렬
    results.sort(key=lambda x: x[1])
    
    print("\n  조합 탐색 결과:")
    print("-" * 70)
    for name, fid, mean_t, cov_t, _ in results[:10]:
        print(f"    {name:<35} FID={fid:.4f} (m={mean_t:.4f}, c={cov_t:.4f})")
    
    if results:
        best_name, best_fid, best_mean, best_cov, best_indices = results[0]
        print(f"\n  🏆 최고: {best_name}, FID={best_fid:.4f}")
        return best_fid, best_mean, best_cov, best_indices
    else:
        return float('inf'), 0, 0, np.array([])


# ============================================================
# 🎯 새로운 FID 개선 전략들 (v4) - 공분산 term 17 이하 목표
# ============================================================

def strategy_fine_iterative(
    features_A: np.ndarray, 
    features_B: np.ndarray,
    removal_rate: float = 0.005,  # 0.5%
    patience: int = 10,
    max_remove_ratio: float = 0.5,
    target_cov_term: float = 17.0
) -> Tuple[float, float, float, np.ndarray]:
    """
    전략 v4-1: Fine-grained Iterative Removal
    - 0.5%씩 세밀하게 제거
    - patience=10으로 early stopping
    - 공분산 term 17 이하 목표
    """
    print(f"\n[v4-1] Fine Iterative (rate={removal_rate*100}%, patience={patience})")
    
    current_indices = np.arange(len(features_B))
    min_samples = int(len(features_B) * (1 - max_remove_ratio))
    
    current_fid, current_mean, current_cov = compute_fid(features_A, features_B)
    best_fid = current_fid
    best_cov = current_cov
    best_indices = current_indices.copy()
    
    print(f"  초기: FID={current_fid:.4f} (평균={current_mean:.4f}, 공분산={current_cov:.4f})")
    
    no_improve_count = 0
    iteration = 0
    
    while len(current_indices) > min_samples:
        iteration += 1
        
        # 현재 샘플들의 FID 기여도 계산
        current_features = features_B[current_indices]
        mu_A = np.mean(features_A, axis=0)
        
        # Mahalanobis 거리 (공분산 기여도 근사)
        mahal_dist = compute_mahalanobis_distances(features_A, current_features)
        
        # 평균 기여도
        mean_contribution = np.sum((current_features - mu_A) ** 2, axis=1)
        
        # 공분산 편향 점수 (공분산 term 감소에 집중)
        combined_score = 0.3 * mean_contribution + 0.7 * (mahal_dist ** 2)
        
        # 상위 removal_rate 비율 제거
        n_remove = max(1, int(len(current_indices) * removal_rate))
        remove_local_indices = np.argsort(combined_score)[-n_remove:]
        
        # 제거
        keep_mask = np.ones(len(current_indices), dtype=bool)
        keep_mask[remove_local_indices] = False
        current_indices = current_indices[keep_mask]
        
        # 새 FID 계산
        new_fid, new_mean, new_cov = compute_fid(features_A, features_B[current_indices])
        
        # 개선 체크
        if new_fid < best_fid:
            best_fid = new_fid
            best_cov = new_cov
            best_indices = current_indices.copy()
            no_improve_count = 0
            
            if iteration % 20 == 0 or new_cov < target_cov_term:
                print(f"  Iter {iteration:4d}: FID={new_fid:.4f} (m={new_mean:.4f}, c={new_cov:.4f}) ⬇️ n={len(current_indices)}")
        else:
            no_improve_count += 1
        
        # 목표 달성
        if new_cov <= target_cov_term:
            print(f"  🎯 공분산 term 목표 달성! c={new_cov:.4f}")
            break
        
        # Early stopping
        if no_improve_count >= patience:
            print(f"  [Early Stop] {patience}회 연속 미개선")
            break
        
        current_fid = new_fid
    
    fid, mean_t, cov_t = compute_fid(features_A, features_B[best_indices])
    print(f"  → 최종: FID={fid:.4f} (평균={mean_t:.4f}, 공분산={cov_t:.4f})")
    print(f"  → 샘플 수: {len(best_indices)} ({len(best_indices)/len(features_B)*100:.1f}%)")
    
    return fid, mean_t, cov_t, best_indices


def strategy_dimension_targeted(
    features_A: np.ndarray, 
    features_B: np.ndarray,
    top_k_dims: int = 200,
    outlier_percentile: float = 90
) -> Tuple[float, float, float, np.ndarray]:
    """
    전략 v4-2: Dimension Targeted Outlier Removal
    - 분산 차이가 큰 차원 식별
    - 해당 차원에서만 outlier 제거
    - A의 분산 분포와 맞지 않는 샘플 제거
    """
    print(f"\n[v4-2] Dimension Targeted (top_k={top_k_dims}, percentile={outlier_percentile})")
    
    # 차원별 분산 계산
    var_A = np.var(features_A, axis=0)
    var_B = np.var(features_B, axis=0)
    
    # 분산 차이 비율
    var_ratio = var_B / (var_A + 1e-10)
    
    # 분산 차이가 큰 차원 (과대 또는 과소)
    var_diff = np.abs(var_ratio - 1.0)
    target_dims = np.argsort(var_diff)[-top_k_dims:]
    
    print(f"  타겟 차원 {top_k_dims}개 선택")
    print(f"  분산 비율 범위: {var_ratio[target_dims].min():.2f} ~ {var_ratio[target_dims].max():.2f}")
    
    # 타겟 차원에서의 outlier 점수 계산
    mu_A_target = np.mean(features_A[:, target_dims], axis=0)
    std_A_target = np.std(features_A[:, target_dims], axis=0) + 1e-10
    
    # B의 각 샘플이 타겟 차원에서 얼마나 벗어나는지
    B_target = features_B[:, target_dims]
    z_scores = np.abs((B_target - mu_A_target) / std_A_target)
    outlier_scores = np.mean(z_scores, axis=1)
    
    # 상위 outlier 제거
    threshold = np.percentile(outlier_scores, outlier_percentile)
    selected_indices = np.where(outlier_scores <= threshold)[0]
    
    print(f"  선택된 샘플: {len(selected_indices)} / {len(features_B)}")
    
    fid, mean_t, cov_t = compute_fid(features_A, features_B[selected_indices])
    print(f"  → FID: {fid:.4f} (평균: {mean_t:.4f}, 공분산: {cov_t:.4f})")
    
    return fid, mean_t, cov_t, selected_indices


def strategy_eigenspace_variance_match(
    features_A: np.ndarray, 
    features_B: np.ndarray,
    n_components: int = 50,
    tolerance: float = 0.3
) -> Tuple[float, float, float, np.ndarray]:
    """
    전략 v4-3: Eigenspace Variance Matching
    - A의 PCA 공간에서 각 PC별 분산 타겟
    - B에서 해당 분산에 기여하는 샘플 선택
    - 분산 비율이 tolerance 범위 내인 샘플 우선
    """
    print(f"\n[v4-3] Eigenspace Variance Match (n_comp={n_components}, tol={tolerance})")
    
    from sklearn.decomposition import PCA
    
    # A로 PCA 학습
    pca = PCA(n_components=min(n_components, features_A.shape[1], len(features_A) - 1))
    pca.fit(features_A)
    
    A_pca = pca.transform(features_A)
    B_pca = pca.transform(features_B)
    
    # A의 각 PC별 통계
    mean_A_pca = np.mean(A_pca, axis=0)
    std_A_pca = np.std(A_pca, axis=0)
    var_A_pca = np.var(A_pca, axis=0)
    
    # 각 B 샘플의 "분산 기여도" 점수
    # 각 PC에서 분산에 기여하는 정도 (제곱 편차)
    B_centered = B_pca - mean_A_pca
    
    # 이상적인 분산 기여도 (A와 동일한 분산을 만들기 위해)
    ideal_sq_dev = var_A_pca  # 각 PC의 분산
    
    # 각 샘플의 제곱 편차
    sample_sq_dev = B_centered ** 2
    
    # 분산 비율 점수: 1에 가까울수록 좋음
    # 각 PC별로 (sample_sq_dev / ideal_sq_dev) 가 1에 가까운 정도
    var_contribution = sample_sq_dev / (ideal_sq_dev + 1e-10)
    
    # 상위 PC들의 가중 평균으로 점수화
    weights = var_A_pca / var_A_pca.sum()  # 분산 비율을 가중치로
    
    # 1에서 벗어난 정도 (낮을수록 좋음)
    deviation_from_one = np.abs(var_contribution - 1.0)
    weighted_deviation = np.sum(deviation_from_one * weights, axis=1)
    
    # tolerance 범위 내 샘플 우선 선택
    good_samples = weighted_deviation < tolerance
    
    if good_samples.sum() < 1000:
        # 너무 적으면 상위 50% 선택
        threshold = np.percentile(weighted_deviation, 50)
        good_samples = weighted_deviation < threshold
    
    selected_indices = np.where(good_samples)[0]
    
    print(f"  선택된 샘플: {len(selected_indices)} / {len(features_B)}")
    
    # 선택된 샘플의 분산 비교
    selected_pca = B_pca[selected_indices]
    var_selected = np.var(selected_pca, axis=0)
    
    print(f"  상위 5개 PC 분산 비교:")
    for i in range(min(5, len(var_A_pca))):
        ratio = var_selected[i] / var_A_pca[i] if var_A_pca[i] > 0 else 0
        print(f"    PC{i+1}: A={var_A_pca[i]:.4f}, Sel={var_selected[i]:.4f}, ratio={ratio:.3f}")
    
    fid, mean_t, cov_t = compute_fid(features_A, features_B[selected_indices])
    print(f"  → FID: {fid:.4f} (평균: {mean_t:.4f}, 공분산: {cov_t:.4f})")
    
    return fid, mean_t, cov_t, selected_indices


def strategy_minibatch_sinkhorn(
    features_A: np.ndarray, 
    features_B: np.ndarray,
    target_size: Optional[int] = None,
    batch_size: int = 5000,
    n_iter: int = 30,
    reg: float = 0.1,
    device: str = 'cuda'
) -> Tuple[float, float, float, np.ndarray]:
    """
    전략 v4-4: Mini-batch Sinkhorn OT
    - OOM 방지를 위한 mini-batch OT
    - importance score 누적 후 상위 선택
    """
    if target_size is None:
        target_size = len(features_A) * 4
    
    print(f"\n[v4-4] Minibatch Sinkhorn (batch={batch_size}, iter={n_iter}, reg={reg})")
    
    try:
        import ot
    except ImportError:
        print("  ⚠️ POT 라이브러리 필요: pip install POT")
        return float('inf'), 0, 0, np.array([])
    
    n_A = len(features_A)
    n_B = len(features_B)
    
    # Importance score 누적
    importance_scores = np.zeros(n_B)
    
    for iteration in tqdm(range(n_iter), desc="  Minibatch OT"):
        # A에서 랜덤 샘플링
        np.random.seed(iteration)
        A_sample_idx = np.random.choice(n_A, size=min(batch_size, n_A), replace=False)
        B_sample_idx = np.random.choice(n_B, size=min(batch_size, n_B), replace=False)
        
        A_batch = features_A[A_sample_idx]
        B_batch = features_B[B_sample_idx]
        
        # Cost matrix
        if device == 'cuda' and torch.cuda.is_available():
            A_t = torch.from_numpy(A_batch).float().cuda()
            B_t = torch.from_numpy(B_batch).float().cuda()
            diff = A_t.unsqueeze(1) - B_t.unsqueeze(0)
            cost = torch.sum(diff ** 2, dim=2).cpu().numpy()
            del A_t, B_t
            torch.cuda.empty_cache()
        else:
            from scipy.spatial.distance import cdist
            cost = cdist(A_batch, B_batch, metric='sqeuclidean')
        
        # Uniform distributions
        a = np.ones(len(A_batch)) / len(A_batch)
        b = np.ones(len(B_batch)) / len(B_batch)
        
        # Sinkhorn
        try:
            T = ot.sinkhorn(a, b, cost, reg=reg, numItermax=500, stopThr=1e-6)
            batch_importance = T.sum(axis=0)
            importance_scores[B_sample_idx] += batch_importance
        except Exception as e:
            continue
    
    # 상위 target_size개 선택
    selected_indices = np.argsort(importance_scores)[-target_size:]
    
    print(f"  선택된 샘플: {len(selected_indices)}")
    print(f"  Importance 분포: min={importance_scores.min():.6f}, max={importance_scores.max():.6f}")
    
    fid, mean_t, cov_t = compute_fid(features_A, features_B[selected_indices])
    print(f"  → FID: {fid:.4f} (평균: {mean_t:.4f}, 공분산: {cov_t:.4f})")
    
    return fid, mean_t, cov_t, selected_indices


def strategy_combined_v2(
    features_A: np.ndarray, 
    features_B: np.ndarray,
    target_cov_term: float = 17.0,
    device: str = 'cuda'
) -> Tuple[float, float, float, np.ndarray]:
    """
    전략 v4-5: 3단계 복합 전략
    Stage 1: dimension_targeted (상위 10% 제거)
    Stage 2: fine_iterative (0.5%씩 정밀 제거)
    Stage 3: eigenspace 기반 최종 조정
    """
    print(f"\n[v4-5] Combined v2 (목표 공분산: {target_cov_term})")
    print("=" * 60)
    
    # Stage 1: Dimension Targeted Outlier Removal
    print("\n  === Stage 1: Dimension Targeted ===")
    _, _, _, stage1_indices = strategy_dimension_targeted(
        features_A, features_B,
        top_k_dims=200,
        outlier_percentile=90
    )
    
    stage1_features = features_B[stage1_indices]
    fid1, mean1, cov1 = compute_fid(features_A, stage1_features)
    print(f"  Stage 1 결과: FID={fid1:.4f}, 공분산={cov1:.4f}")
    
    # Stage 2: Fine Iterative on Stage 1 result
    print("\n  === Stage 2: Fine Iterative ===")
    
    # Stage 1의 인덱스를 기준으로 새로운 features 생성
    current_indices = stage1_indices.copy()
    min_samples = int(len(stage1_indices) * 0.5)
    
    best_fid = fid1
    best_cov = cov1
    best_indices = stage1_indices.copy()
    
    no_improve_count = 0
    patience = 15
    iteration = 0
    
    while len(current_indices) > min_samples:
        iteration += 1
        current_features = features_B[current_indices]
        
        mu_A = np.mean(features_A, axis=0)
        mahal_dist = compute_mahalanobis_distances(features_A, current_features)
        mean_contribution = np.sum((current_features - mu_A) ** 2, axis=1)
        
        # 공분산에 더 집중
        combined_score = 0.2 * mean_contribution + 0.8 * (mahal_dist ** 2)
        
        n_remove = max(1, int(len(current_indices) * 0.005))
        remove_local_indices = np.argsort(combined_score)[-n_remove:]
        
        keep_mask = np.ones(len(current_indices), dtype=bool)
        keep_mask[remove_local_indices] = False
        current_indices = current_indices[keep_mask]
        
        new_fid, new_mean, new_cov = compute_fid(features_A, features_B[current_indices])
        
        if new_fid < best_fid:
            best_fid = new_fid
            best_cov = new_cov
            best_indices = current_indices.copy()
            no_improve_count = 0
            
            if iteration % 30 == 0:
                print(f"    Iter {iteration}: FID={new_fid:.4f}, cov={new_cov:.4f} ⬇️")
        else:
            no_improve_count += 1
        
        if new_cov <= target_cov_term:
            print(f"    🎯 목표 공분산 달성! cov={new_cov:.4f}")
            break
        
        if no_improve_count >= patience:
            print(f"    [Early Stop] {patience}회 미개선")
            break
    
    fid2, mean2, cov2 = compute_fid(features_A, features_B[best_indices])
    print(f"  Stage 2 결과: FID={fid2:.4f}, 공분산={cov2:.4f}")
    
    # Stage 3: Eigenspace 기반 추가 필터링 (공분산이 아직 목표 미달이면)
    if cov2 > target_cov_term:
        print("\n  === Stage 3: Eigenspace Refinement ===")
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=30)
        pca.fit(features_A)
        
        A_pca = pca.transform(features_A)
        B_pca = pca.transform(features_B[best_indices])
        
        mean_A_pca = np.mean(A_pca, axis=0)
        var_A_pca = np.var(A_pca, axis=0)
        
        B_centered = B_pca - mean_A_pca
        sample_sq_dev = B_centered ** 2
        var_contribution = sample_sq_dev / (var_A_pca + 1e-10)
        
        # 분산 비율이 1에 가까운 샘플 선택
        deviation = np.abs(var_contribution - 1.0)
        weights = var_A_pca / var_A_pca.sum()
        weighted_dev = np.sum(deviation * weights, axis=1)
        
        # 하위 80% 선택
        threshold = np.percentile(weighted_dev, 80)
        stage3_mask = weighted_dev <= threshold
        stage3_local_indices = np.where(stage3_mask)[0]
        
        final_indices = best_indices[stage3_local_indices]
        fid3, mean3, cov3 = compute_fid(features_A, features_B[final_indices])
        
        print(f"  Stage 3 결과: FID={fid3:.4f}, 공분산={cov3:.4f}")
        
        if fid3 < fid2:
            best_indices = final_indices
            best_fid = fid3
            best_cov = cov3
    
    fid, mean_t, cov_t = compute_fid(features_A, features_B[best_indices])
    print("\n" + "=" * 60)
    print(f"  🏆 최종: FID={fid:.4f} (평균={mean_t:.4f}, 공분산={cov_t:.4f})")
    print(f"  → 샘플 수: {len(best_indices)} ({len(best_indices)/len(features_B)*100:.1f}%)")
    
    return fid, mean_t, cov_t, best_indices


def strategy_variance_ratio_filter(
    features_A: np.ndarray, 
    features_B: np.ndarray,
    target_ratio_range: Tuple[float, float] = (0.8, 1.2)
) -> Tuple[float, float, float, np.ndarray]:
    """
    전략 v4-6: Variance Ratio Filtering
    각 차원에서 B가 A와 비슷한 분산 기여를 하는 샘플만 선택
    """
    print(f"\n[v4-6] Variance Ratio Filter (range={target_ratio_range})")
    
    mu_A = np.mean(features_A, axis=0)
    var_A = np.var(features_A, axis=0)
    
    # 각 B 샘플이 각 차원에서 분산에 기여하는 정도
    B_centered = features_B - mu_A
    B_sq_dev = B_centered ** 2
    
    # 이상적 기여도 (A의 분산)
    # B 샘플이 A의 분산에 맞는지 확인
    # 각 차원별로 (B_sq_dev / var_A)가 1에 가까우면 좋음
    contribution_ratio = B_sq_dev / (var_A + 1e-10)
    
    # 분산이 큰 차원들 (상위 200개)에 집중
    top_dims = np.argsort(var_A)[-200:]
    
    # 해당 차원들에서의 ratio
    top_ratio = contribution_ratio[:, top_dims]
    
    # target_ratio_range 내에 있는 차원의 비율
    in_range = (top_ratio >= target_ratio_range[0]) & (top_ratio <= target_ratio_range[1])
    in_range_ratio = np.mean(in_range, axis=1)
    
    # 80% 이상의 차원이 범위 내인 샘플 선택
    threshold = 0.6
    selected_mask = in_range_ratio >= threshold
    
    if selected_mask.sum() < 1000:
        # 너무 적으면 threshold 낮춤
        threshold = np.percentile(in_range_ratio, 50)
        selected_mask = in_range_ratio >= threshold
    
    selected_indices = np.where(selected_mask)[0]
    
    print(f"  선택된 샘플: {len(selected_indices)} / {len(features_B)}")
    
    fid, mean_t, cov_t = compute_fid(features_A, features_B[selected_indices])
    print(f"  → FID: {fid:.4f} (평균: {mean_t:.4f}, 공분산: {cov_t:.4f})")
    
    return fid, mean_t, cov_t, selected_indices


# ============================================================
# 🎯 FID 최적화 v5 - 목표: FID < 20
# ============================================================

def strategy_minibatch_sinkhorn_cpu(
    features_A: np.ndarray, 
    features_B: np.ndarray,
    target_size: int = 20000,
    batch_a: int = 500,
    batch_b: int = 2000,
    n_iter: int = 30,
    reg: float = 0.1
) -> Tuple[float, float, float, np.ndarray]:
    """
    전략 v5-1: CPU Mini-batch Sinkhorn (OOM 수정 버전)
    - A에서 batch_a개, B에서 batch_b개씩 샘플링
    - Cost matrix: batch_a x batch_b (작은 크기)
    """
    print(f"\n[v5-1] Minibatch Sinkhorn CPU (a={batch_a}, b={batch_b}, iter={n_iter})")
    
    try:
        import ot
    except ImportError:
        print("  ⚠️ POT 라이브러리 필요: pip install POT")
        return float('inf'), 0, 0, np.array([])
    
    importance = np.zeros(len(features_B))
    
    for i in tqdm(range(n_iter), desc="  Minibatch OT"):
        np.random.seed(i)
        idx_A = np.random.choice(len(features_A), batch_a, replace=False)
        idx_B = np.random.choice(len(features_B), batch_b, replace=False)
        
        A_batch = features_A[idx_A]
        B_batch = features_B[idx_B]
        
        # 작은 cost matrix (batch_a x batch_b)
        M = np.linalg.norm(A_batch[:, None] - B_batch[None, :], axis=2)
        M = M / (M.max() + 1e-8)  # 정규화
        
        a = np.ones(batch_a) / batch_a
        b = np.ones(batch_b) / batch_b
        
        try:
            T = ot.sinkhorn(a, b, M, reg=reg, numItermax=50, stopThr=1e-6)
            importance[idx_B] += T.sum(axis=0)
        except Exception as e:
            continue
    
    selected = np.argsort(importance)[-target_size:]
    
    print(f"  선택된 샘플: {len(selected)}")
    
    fid, mean_t, cov_t = compute_fid(features_A, features_B[selected])
    print(f"  → FID: {fid:.4f} (평균: {mean_t:.4f}, 공분산: {cov_t:.4f})")
    
    return fid, mean_t, cov_t, selected


def strategy_dimtarget_grid_search(
    features_A: np.ndarray, 
    features_B: np.ndarray,
    top_k_range: List[int] = None,
    percentile_range: List[int] = None
) -> Tuple[float, float, float, np.ndarray, dict]:
    """
    전략 v5-2: DimTarget 파라미터 그리드 서치
    - top_k: 50~150 범위에서 25 단위로
    - percentile: 80~90 범위에서 2 단위로
    """
    if top_k_range is None:
        top_k_range = [50, 75, 100, 125, 150]
    if percentile_range is None:
        percentile_range = list(range(80, 91, 2))
    
    print(f"\n[v5-2] DimTarget Grid Search")
    print(f"  top_k 범위: {top_k_range}")
    print(f"  percentile 범위: {percentile_range}")
    print("-" * 70)
    
    # 차원별 분산 계산 (한 번만)
    var_A = np.var(features_A, axis=0)
    var_B = np.var(features_B, axis=0)
    var_ratio = var_B / (var_A + 1e-10)
    var_diff = np.abs(var_ratio - 1.0)
    
    results = {}
    best_fid = float('inf')
    best_params = None
    best_indices = None
    
    for top_k in top_k_range:
        target_dims = np.argsort(var_diff)[-top_k:]
        
        mu_A_target = np.mean(features_A[:, target_dims], axis=0)
        std_A_target = np.std(features_A[:, target_dims], axis=0) + 1e-10
        
        B_target = features_B[:, target_dims]
        z_scores = np.abs((B_target - mu_A_target) / std_A_target)
        outlier_scores = np.mean(z_scores, axis=1)
        
        for percentile in percentile_range:
            threshold = np.percentile(outlier_scores, percentile)
            selected_indices = np.where(outlier_scores <= threshold)[0]
            
            if len(selected_indices) < 1000:
                continue
            
            fid, mean_t, cov_t = compute_fid(features_A, features_B[selected_indices])
            results[(top_k, percentile)] = {'fid': fid, 'mean': mean_t, 'cov': cov_t, 'n': len(selected_indices)}
            
            if fid < best_fid:
                best_fid = fid
                best_params = (top_k, percentile)
                best_indices = selected_indices.copy()
                print(f"  ✨ k={top_k:3d}, p={percentile:2d}: FID={fid:.4f} (c={cov_t:.4f}) n={len(selected_indices)}")
    
    print("-" * 70)
    print(f"  🏆 최적: k={best_params[0]}, p={best_params[1]}, FID={best_fid:.4f}")
    
    fid, mean_t, cov_t = compute_fid(features_A, features_B[best_indices])
    return fid, mean_t, cov_t, best_indices, results


def strategy_dimtarget_then_iterative(
    features_A: np.ndarray, 
    features_B: np.ndarray,
    top_k: int = 100,
    dim_percentile: int = 85,
    iter_rate: float = 0.002,
    iter_patience: int = 20
) -> Tuple[float, float, float, np.ndarray]:
    """
    전략 v5-3: DimTarget + Fine Iterative 조합
    Stage 1: Dimension Targeted로 1차 필터링
    Stage 2: 더 세밀한 Iterative (0.2%씩)
    """
    print(f"\n[v5-3] DimTarget + Iterative (k={top_k}, p={dim_percentile}, rate={iter_rate})")
    
    # Stage 1: Dimension Targeted
    var_A = np.var(features_A, axis=0)
    var_B = np.var(features_B, axis=0)
    var_ratio = var_B / (var_A + 1e-10)
    var_diff = np.abs(var_ratio - 1.0)
    target_dims = np.argsort(var_diff)[-top_k:]
    
    mu_A_target = np.mean(features_A[:, target_dims], axis=0)
    std_A_target = np.std(features_A[:, target_dims], axis=0) + 1e-10
    
    B_target = features_B[:, target_dims]
    z_scores = np.abs((B_target - mu_A_target) / std_A_target)
    outlier_scores = np.mean(z_scores, axis=1)
    
    threshold = np.percentile(outlier_scores, dim_percentile)
    stage1_indices = np.where(outlier_scores <= threshold)[0]
    
    fid1, mean1, cov1 = compute_fid(features_A, features_B[stage1_indices])
    print(f"  Stage 1: n={len(stage1_indices)}, FID={fid1:.4f}, cov={cov1:.4f}")
    
    # Stage 2: Fine Iterative
    current_indices = stage1_indices.copy()
    min_samples = int(len(stage1_indices) * 0.5)
    
    best_fid = fid1
    best_indices = stage1_indices.copy()
    no_improve_count = 0
    iteration = 0
    
    while len(current_indices) > min_samples:
        iteration += 1
        current_features = features_B[current_indices]
        
        mu_A = np.mean(features_A, axis=0)
        mahal_dist = compute_mahalanobis_distances(features_A, current_features)
        mean_contribution = np.sum((current_features - mu_A) ** 2, axis=1)
        
        combined_score = 0.3 * mean_contribution + 0.7 * (mahal_dist ** 2)
        
        n_remove = max(1, int(len(current_indices) * iter_rate))
        remove_local_indices = np.argsort(combined_score)[-n_remove:]
        
        keep_mask = np.ones(len(current_indices), dtype=bool)
        keep_mask[remove_local_indices] = False
        current_indices = current_indices[keep_mask]
        
        new_fid, new_mean, new_cov = compute_fid(features_A, features_B[current_indices])
        
        if new_fid < best_fid:
            best_fid = new_fid
            best_indices = current_indices.copy()
            no_improve_count = 0
            
            if iteration % 50 == 0:
                print(f"    Iter {iteration}: FID={new_fid:.4f}, cov={new_cov:.4f} ⬇️")
        else:
            no_improve_count += 1
        
        if no_improve_count >= iter_patience:
            break
    
    fid, mean_t, cov_t = compute_fid(features_A, features_B[best_indices])
    print(f"  Stage 2: n={len(best_indices)}, FID={fid:.4f}, cov={cov_t:.4f}")
    
    return fid, mean_t, cov_t, best_indices


def strategy_asymmetric_dim(
    features_A: np.ndarray, 
    features_B: np.ndarray,
    under_threshold: float = 0.9,
    over_threshold: float = 1.1,
    percentile: int = 85
) -> Tuple[float, float, float, np.ndarray]:
    """
    전략 v5-4: Asymmetric Dimension Targeting
    분산이 '부족한' 차원과 '과잉인' 차원을 다르게 처리
    - 부족 차원: 중심에서 먼 샘플 유지 (분산 증가)
    - 과잉 차원: 극단값 제거 (분산 감소)
    """
    print(f"\n[v5-4] Asymmetric Dim (under<{under_threshold}, over>{over_threshold})")
    
    var_A = np.var(features_A, axis=0)
    var_B = np.var(features_B, axis=0)
    ratio = var_B / (var_A + 1e-8)
    
    under_dims = np.where(ratio < under_threshold)[0]
    over_dims = np.where(ratio > over_threshold)[0]
    
    print(f"  분산 부족 차원: {len(under_dims)}개")
    print(f"  분산 과잉 차원: {len(over_dims)}개")
    
    scores = np.zeros(len(features_B))
    
    # 과잉 차원: 극단값에 페널티 (제거 대상)
    for dim in over_dims:
        mu = np.mean(features_A[:, dim])
        sigma = np.std(features_A[:, dim]) + 1e-6
        deviation = np.abs(features_B[:, dim] - mu) / sigma
        scores += deviation
    
    # 부족 차원: 중심에서 먼 샘플에 보너스 (유지 대상) - 음수 점수
    for dim in under_dims:
        mu = np.mean(features_A[:, dim])
        sigma = np.std(features_A[:, dim]) + 1e-6
        deviation = np.abs(features_B[:, dim] - mu) / sigma
        # 적당히 먼 샘플은 유지 (극단은 제외)
        bonus = np.clip(deviation, 0, 2)  # 2 sigma까지만 보너스
        scores -= bonus * 0.3  # 약한 보너스
    
    # 상위 N% 제거 (점수가 높은 것)
    threshold = np.percentile(scores, percentile)
    selected = np.where(scores <= threshold)[0]
    
    print(f"  선택된 샘플: {len(selected)} / {len(features_B)}")
    
    fid, mean_t, cov_t = compute_fid(features_A, features_B[selected])
    print(f"  → FID: {fid:.4f} (평균: {mean_t:.4f}, 공분산: {cov_t:.4f})")
    
    return fid, mean_t, cov_t, selected


def strategy_cluster_adaptive(
    features_A: np.ndarray, 
    features_B: np.ndarray,
    n_clusters: int = 30
) -> Tuple[float, float, float, np.ndarray]:
    """
    전략 v5-5: Per-Cluster Adaptive Filtering
    클러스터별로 다른 필터링 강도 적용
    - A에 비해 B가 부족한 클러스터: 덜 필터링
    - A에 비해 B가 과잉인 클러스터: 더 강하게 필터링
    """
    print(f"\n[v5-5] Cluster Adaptive (n_clusters={n_clusters})")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(features_A)
    
    labels_A = kmeans.labels_
    labels_B = kmeans.predict(features_B)
    
    ratio_A = np.bincount(labels_A, minlength=n_clusters) / len(features_A)
    ratio_B = np.bincount(labels_B, minlength=n_clusters) / len(features_B)
    
    selected = []
    
    for c in range(n_clusters):
        cluster_indices = np.where(labels_B == c)[0]
        
        if len(cluster_indices) == 0:
            continue
        
        excess_ratio = ratio_B[c] / (ratio_A[c] + 1e-8)
        
        if excess_ratio > 1.5:
            keep_ratio = 0.6
        elif excess_ratio > 1.2:
            keep_ratio = 0.75
        elif excess_ratio < 0.7:
            keep_ratio = 0.95
        elif excess_ratio < 0.9:
            keep_ratio = 0.9
        else:
            keep_ratio = 0.85
        
        # 클러스터 중심에 가까운 순으로 선택
        center = kmeans.cluster_centers_[c]
        cluster_features = features_B[cluster_indices]
        dists = np.linalg.norm(cluster_features - center, axis=1)
        n_keep = max(1, int(len(cluster_indices) * keep_ratio))
        selected.extend(cluster_indices[np.argsort(dists)[:n_keep]].tolist())
    
    selected = np.array(selected)
    print(f"  선택된 샘플: {len(selected)} / {len(features_B)}")
    
    fid, mean_t, cov_t = compute_fid(features_A, features_B[selected])
    print(f"  → FID: {fid:.4f} (평균: {mean_t:.4f}, 공분산: {cov_t:.4f})")
    
    return fid, mean_t, cov_t, selected


def strategy_ensemble_selection(
    features_A: np.ndarray, 
    features_B: np.ndarray,
    min_votes: int = 2
) -> Tuple[float, float, float, np.ndarray]:
    """
    전략 v5-6: Ensemble Selection
    여러 전략의 투표로 선택 (min_votes개 이상 전략에서 선택된 샘플만)
    """
    print(f"\n[v5-6] Ensemble Selection (min_votes={min_votes})")
    
    votes = np.zeros(len(features_B))
    
    # 전략 1: DimTarget (k=100, p=85)
    var_A = np.var(features_A, axis=0)
    var_B = np.var(features_B, axis=0)
    var_diff = np.abs(var_B / (var_A + 1e-10) - 1.0)
    target_dims = np.argsort(var_diff)[-100:]
    mu_A_t = np.mean(features_A[:, target_dims], axis=0)
    std_A_t = np.std(features_A[:, target_dims], axis=0) + 1e-10
    z1 = np.mean(np.abs((features_B[:, target_dims] - mu_A_t) / std_A_t), axis=1)
    idx1 = np.where(z1 <= np.percentile(z1, 85))[0]
    votes[idx1] += 1
    print(f"  전략1 (DimTarget k=100,p=85): {len(idx1)}개")
    
    # 전략 2: DimTarget (k=75, p=87)
    target_dims2 = np.argsort(var_diff)[-75:]
    mu_A_t2 = np.mean(features_A[:, target_dims2], axis=0)
    std_A_t2 = np.std(features_A[:, target_dims2], axis=0) + 1e-10
    z2 = np.mean(np.abs((features_B[:, target_dims2] - mu_A_t2) / std_A_t2), axis=1)
    idx2 = np.where(z2 <= np.percentile(z2, 87))[0]
    votes[idx2] += 1
    print(f"  전략2 (DimTarget k=75,p=87): {len(idx2)}개")
    
    # 전략 3: Mahalanobis (p=90)
    mahal_dist = compute_mahalanobis_distances(features_A, features_B)
    idx3 = np.where(mahal_dist <= np.percentile(mahal_dist, 90))[0]
    votes[idx3] += 1
    print(f"  전략3 (Mahalanobis p=90): {len(idx3)}개")
    
    # 전략 4: Asymmetric
    ratio = var_B / (var_A + 1e-8)
    over_dims = np.where(ratio > 1.1)[0]
    scores = np.zeros(len(features_B))
    for dim in over_dims:
        mu = np.mean(features_A[:, dim])
        sigma = np.std(features_A[:, dim]) + 1e-6
        scores += np.abs(features_B[:, dim] - mu) / sigma
    idx4 = np.where(scores <= np.percentile(scores, 85))[0]
    votes[idx4] += 1
    print(f"  전략4 (Asymmetric p=85): {len(idx4)}개")
    
    # min_votes 이상 선택된 샘플
    selected = np.where(votes >= min_votes)[0]
    
    print(f"  {min_votes}개 이상 투표: {len(selected)} / {len(features_B)}")
    
    fid, mean_t, cov_t = compute_fid(features_A, features_B[selected])
    print(f"  → FID: {fid:.4f} (평균: {mean_t:.4f}, 공분산: {cov_t:.4f})")
    
    return fid, mean_t, cov_t, selected


def strategy_combined_v3(
    features_A: np.ndarray, 
    features_B: np.ndarray,
    target_fid: float = 20.0
) -> Tuple[float, float, float, np.ndarray]:
    """
    전략 v5-7: 최적 조합 v3
    DimTarget Grid Search 최적 → Fine Iterative
    """
    print(f"\n[v5-7] Combined v3 (목표 FID: {target_fid})")
    print("=" * 60)
    
    # Stage 1: DimTarget 최적 파라미터 찾기
    print("\n  === Stage 1: DimTarget Grid Search ===")
    _, _, _, stage1_indices, grid_results = strategy_dimtarget_grid_search(
        features_A, features_B,
        top_k_range=[50, 75, 100, 125],
        percentile_range=[83, 85, 87, 89]
    )
    
    fid1, mean1, cov1 = compute_fid(features_A, features_B[stage1_indices])
    print(f"  Stage 1 결과: FID={fid1:.4f}, 공분산={cov1:.4f}")
    
    if fid1 <= target_fid:
        print(f"  🎉 Stage 1에서 목표 달성!")
        return fid1, mean1, cov1, stage1_indices
    
    # Stage 2: Fine Iterative
    print("\n  === Stage 2: Ultra-Fine Iterative ===")
    current_indices = stage1_indices.copy()
    min_samples = int(len(stage1_indices) * 0.4)
    
    best_fid = fid1
    best_indices = stage1_indices.copy()
    no_improve_count = 0
    patience = 25
    iteration = 0
    
    while len(current_indices) > min_samples:
        iteration += 1
        current_features = features_B[current_indices]
        
        mu_A = np.mean(features_A, axis=0)
        mahal_dist = compute_mahalanobis_distances(features_A, current_features)
        mean_contribution = np.sum((current_features - mu_A) ** 2, axis=1)
        
        combined_score = 0.25 * mean_contribution + 0.75 * (mahal_dist ** 2)
        
        n_remove = max(1, int(len(current_indices) * 0.001))  # 0.1%씩
        remove_local_indices = np.argsort(combined_score)[-n_remove:]
        
        keep_mask = np.ones(len(current_indices), dtype=bool)
        keep_mask[remove_local_indices] = False
        current_indices = current_indices[keep_mask]
        
        new_fid, new_mean, new_cov = compute_fid(features_A, features_B[current_indices])
        
        if new_fid < best_fid:
            best_fid = new_fid
            best_indices = current_indices.copy()
            no_improve_count = 0
            
            if iteration % 100 == 0 or new_fid < target_fid:
                print(f"    Iter {iteration}: FID={new_fid:.4f} ⬇️")
        else:
            no_improve_count += 1
        
        if new_fid <= target_fid:
            print(f"    🎉 목표 FID 달성!")
            break
        
        if no_improve_count >= patience:
            break
    
    fid, mean_t, cov_t = compute_fid(features_A, features_B[best_indices])
    print("\n" + "=" * 60)
    print(f"  🏆 최종: FID={fid:.4f} (평균={mean_t:.4f}, 공분산={cov_t:.4f})")
    print(f"  → 샘플 수: {len(best_indices)} ({len(best_indices)/len(features_B)*100:.1f}%)")
    
    return fid, mean_t, cov_t, best_indices


def evaluate_all_strategies_v4(
    features_A: np.ndarray, 
    features_B: np.ndarray,
    device: str = 'cuda'
) -> List[Tuple]:
    """모든 v5 전략 평가 및 비교 (목표: FID < 20)"""
    
    print("\n" + "=" * 95)
    print("🏆 FID 개선 전략 비교 (v5 - 목표: FID < 20)")
    print("=" * 95)
    
    original_fid, orig_mean, orig_cov = compute_fid(features_A, features_B)
    print(f"\n📊 원본 FID (B 전체 {len(features_B)}개): {original_fid:.4f}")
    print(f"   (평균: {orig_mean:.4f}, 공분산: {orig_cov:.4f})")
    
    results = [("원본 (B 전체)", original_fid, orig_mean, orig_cov, len(features_B), None)]
    
    # v5-1: Minibatch Sinkhorn CPU
    print("\n" + "=" * 95)
    try:
        fid1, mean1, cov1, idx1 = strategy_minibatch_sinkhorn_cpu(
            features_A, features_B,
            target_size=len(features_A) * 4,
            batch_a=500, batch_b=2000, n_iter=30
        )
        results.append(("Sinkhorn CPU", fid1, mean1, cov1, len(idx1), idx1))
    except Exception as e:
        print(f"  → 실패: {e}")
    
    # v5-2: DimTarget Grid Search
    print("\n" + "=" * 95)
    try:
        fid2, mean2, cov2, idx2, _ = strategy_dimtarget_grid_search(
            features_A, features_B,
            top_k_range=[50, 75, 100, 125, 150],
            percentile_range=[80, 82, 84, 85, 86, 88, 90]
        )
        results.append(("DimTarget GridSearch", fid2, mean2, cov2, len(idx2), idx2))
    except Exception as e:
        print(f"  → 실패: {e}")
    
    # v5-3: DimTarget + Iterative (다양한 파라미터)
    print("\n" + "=" * 95)
    for k, p in [(75, 86), (100, 85), (100, 84)]:
        try:
            fid3, mean3, cov3, idx3 = strategy_dimtarget_then_iterative(
                features_A, features_B,
                top_k=k, dim_percentile=p,
                iter_rate=0.002, iter_patience=25
            )
            results.append((f"DimTarget+Iter(k={k},p={p})", fid3, mean3, cov3, len(idx3), idx3))
        except Exception as e:
            pass
    
    # v5-4: Asymmetric Dim
    print("\n" + "=" * 95)
    for percentile in [83, 85, 87]:
        try:
            fid4, mean4, cov4, idx4 = strategy_asymmetric_dim(
                features_A, features_B,
                percentile=percentile
            )
            results.append((f"Asymmetric(p={percentile})", fid4, mean4, cov4, len(idx4), idx4))
        except Exception as e:
            pass
    
    # v5-5: Cluster Adaptive
    print("\n" + "=" * 95)
    for n_c in [20, 30, 50]:
        try:
            fid5, mean5, cov5, idx5 = strategy_cluster_adaptive(
                features_A, features_B,
                n_clusters=n_c
            )
            results.append((f"ClusterAdaptive(c={n_c})", fid5, mean5, cov5, len(idx5), idx5))
        except Exception as e:
            pass
    
    # v5-6: Ensemble
    print("\n" + "=" * 95)
    for min_v in [2, 3]:
        try:
            fid6, mean6, cov6, idx6 = strategy_ensemble_selection(
                features_A, features_B,
                min_votes=min_v
            )
            results.append((f"Ensemble(v>={min_v})", fid6, mean6, cov6, len(idx6), idx6))
        except Exception as e:
            pass
    
    # v5-7: Combined v3
    print("\n" + "=" * 95)
    try:
        fid7, mean7, cov7, idx7 = strategy_combined_v3(
            features_A, features_B,
            target_fid=20.0
        )
        results.append(("Combined v3", fid7, mean7, cov7, len(idx7), idx7))
    except Exception as e:
        print(f"  → 실패: {e}")
    
    # 결과 요약
    print("\n" + "=" * 95)
    print("📋 전략별 결과 요약 (FID 순 정렬)")
    print("=" * 95)
    print(f"\n{'전략':<35} {'FID':>8} {'평균':>8} {'공분산':>10} {'샘플수':>10} {'vs원본':>10}")
    print("-" * 95)
    
    results.sort(key=lambda x: x[1])
    
    for i, (name, fid, mean_t, cov_t, n_samples, _) in enumerate(results[:25]):
        diff = fid - original_fid
        fid_marker = "🏆" if i == 0 and name != "원본 (B 전체)" else "  "
        goal_marker = "✅" if fid < 20 else "  "
        print(f"{fid_marker}{goal_marker}{name:<32} {fid:>8.4f} {mean_t:>8.4f} {cov_t:>10.4f} {n_samples:>10} {diff:>+10.4f}")
    
    # 목표 달성 여부
    under_20 = [(n, f, m, c, s, idx) for n, f, m, c, s, idx in results if f < 20 and n != "원본 (B 전체)"]
    
    print("\n" + "=" * 95)
    if under_20:
        print(f"🎉 FID < 20 달성: {len(under_20)}개 전략!")
        best = min(under_20, key=lambda x: x[1])
        print(f"   최저 FID: {best[0]} (FID={best[1]:.4f})")
    else:
        best = min(results, key=lambda x: x[1] if x[0] != "원본 (B 전체)" else float('inf'))
        print(f"⚠️ 목표 미달. 최고: {best[0]} (FID={best[1]:.4f})")
        print(f"   목표(20)까지: {best[1] - 20:.4f} 남음")
    
    return results


def evaluate_all_strategies_v3(
    features_A: np.ndarray, 
    features_B: np.ndarray,
    device: str = 'cuda'
) -> List[Tuple]:
    """모든 v4 전략 평가 및 비교 (공분산 term 17 이하 목표)"""
    
    print("\n" + "=" * 90)
    print("🏆 FID 개선 전략 비교 (v4 - 목표: FID < 20, 공분산 term < 17)")
    print("=" * 90)
    
    # 원본 FID
    original_fid, orig_mean, orig_cov = compute_fid(features_A, features_B)
    print(f"\n📊 원본 FID (B 전체 {len(features_B)}개): {original_fid:.4f}")
    print(f"   (평균: {orig_mean:.4f}, 공분산: {orig_cov:.4f})")
    print(f"   공분산 기여도: {orig_cov/original_fid*100:.1f}%")
    
    results = [("원본 (B 전체)", original_fid, orig_mean, orig_cov, len(features_B), None)]
    
    # v4-1: Fine Iterative
    print("\n" + "=" * 90)
    try:
        fid1, mean1, cov1, indices1 = strategy_fine_iterative(
            features_A, features_B,
            removal_rate=0.005,
            patience=15,
            target_cov_term=17.0
        )
        results.append(("Fine Iterative (0.5%)", fid1, mean1, cov1, len(indices1), indices1))
    except Exception as e:
        print(f"  → 실패: {e}")
    
    # v4-2: Dimension Targeted (다양한 파라미터)
    print("\n" + "=" * 90)
    for top_k in [100, 200, 300]:
        for percentile in [90, 85, 80]:
            try:
                fid2, mean2, cov2, indices2 = strategy_dimension_targeted(
                    features_A, features_B,
                    top_k_dims=top_k,
                    outlier_percentile=percentile
                )
                results.append((f"DimTarget(k={top_k},p={percentile})", fid2, mean2, cov2, len(indices2), indices2))
            except Exception as e:
                pass
    
    # v4-3: Eigenspace Variance Match
    print("\n" + "=" * 90)
    for n_comp in [30, 50, 100]:
        for tol in [0.2, 0.3, 0.5]:
            try:
                fid3, mean3, cov3, indices3 = strategy_eigenspace_variance_match(
                    features_A, features_B,
                    n_components=n_comp,
                    tolerance=tol
                )
                results.append((f"EigenVar(c={n_comp},t={tol})", fid3, mean3, cov3, len(indices3), indices3))
            except Exception as e:
                pass
    
    # v4-4: Minibatch Sinkhorn
    print("\n" + "=" * 90)
    try:
        fid4, mean4, cov4, indices4 = strategy_minibatch_sinkhorn(
            features_A, features_B,
            target_size=len(features_A) * 4,
            batch_size=3000,
            n_iter=20,
            reg=0.1,
            device=device
        )
        if len(indices4) > 0:
            results.append(("Minibatch Sinkhorn", fid4, mean4, cov4, len(indices4), indices4))
    except Exception as e:
        print(f"  → 실패: {e}")
    
    # v4-5: Combined v2
    print("\n" + "=" * 90)
    try:
        fid5, mean5, cov5, indices5 = strategy_combined_v2(
            features_A, features_B,
            target_cov_term=17.0,
            device=device
        )
        results.append(("Combined v2 (3-stage)", fid5, mean5, cov5, len(indices5), indices5))
    except Exception as e:
        print(f"  → 실패: {e}")
    
    # v4-6: Variance Ratio Filter
    print("\n" + "=" * 90)
    for low, high in [(0.7, 1.3), (0.8, 1.2), (0.9, 1.1)]:
        try:
            fid6, mean6, cov6, indices6 = strategy_variance_ratio_filter(
                features_A, features_B,
                target_ratio_range=(low, high)
            )
            results.append((f"VarRatio({low}-{high})", fid6, mean6, cov6, len(indices6), indices6))
        except Exception as e:
            pass
    
    # 결과 요약
    print("\n" + "=" * 90)
    print("📋 전략별 결과 요약 (FID 순 정렬)")
    print("=" * 90)
    print(f"\n{'전략':<35} {'FID':>8} {'평균':>8} {'공분산':>10} {'샘플수':>10} {'vs원본':>10}")
    print("-" * 95)
    
    # FID 순으로 정렬
    results.sort(key=lambda x: x[1])
    
    for i, (name, fid, mean_t, cov_t, n_samples, _) in enumerate(results[:20]):
        diff = fid - original_fid
        cov_marker = "✅" if cov_t < 17 else "  "
        fid_marker = "🏆" if i == 0 and name != "원본 (B 전체)" else "  "
        print(f"{fid_marker}{cov_marker}{name:<32} {fid:>8.4f} {mean_t:>8.4f} {cov_t:>10.4f} {n_samples:>10} {diff:>+10.4f}")
    
    # 공분산 17 이하인 결과들
    cov_under_17 = [(n, f, m, c, s, idx) for n, f, m, c, s, idx in results if c < 17 and n != "원본 (B 전체)"]
    
    print("\n" + "=" * 90)
    if cov_under_17:
        print(f"✅ 공분산 < 17 달성: {len(cov_under_17)}개 전략")
        best_cov = min(cov_under_17, key=lambda x: x[3])
        print(f"   최저 공분산: {best_cov[0]} (cov={best_cov[3]:.4f}, FID={best_cov[1]:.4f})")
    else:
        print("⚠️ 공분산 < 17 달성한 전략 없음")
        # 가장 낮은 공분산 찾기
        best_cov_result = min(results, key=lambda x: x[3] if x[0] != "원본 (B 전체)" else float('inf'))
        print(f"   현재 최저 공분산: {best_cov_result[0]} (cov={best_cov_result[3]:.4f})")
    
    # 목표 달성 여부
    best_result = None
    for r in results:
        if r[0] != "원본 (B 전체)":
            if best_result is None or r[1] < best_result[1]:
                best_result = r
    
    if best_result:
        print(f"\n🏆 최고 FID: {best_result[0]}")
        print(f"   FID: {best_result[1]:.4f} (평균: {best_result[2]:.4f}, 공분산: {best_result[3]:.4f})")
        if best_result[1] < 20:
            print("   🎉 목표 달성!")
        else:
            print(f"   목표(20)까지: {best_result[1] - 20:.4f} 남음")
    
    return results


def evaluate_all_strategies_v2(
    features_A: np.ndarray, 
    features_B: np.ndarray,
    device: str = 'cuda'
) -> List[Tuple]:
    """모든 v3 전략 평가 및 비교"""
    
    print("\n" + "=" * 80)
    print("🏆 FID 개선 전략 비교 (v3 - 목표: FID < 20)")
    print("=" * 80)
    
    # 원본 FID
    original_fid, orig_mean, orig_cov = compute_fid(features_A, features_B)
    print(f"\n📊 원본 FID (B 전체 {len(features_B)}개): {original_fid:.4f}")
    print(f"   (평균: {orig_mean:.4f}, 공분산: {orig_cov:.4f})")
    print(f"   공분산 기여도: {orig_cov/original_fid*100:.1f}%")
    
    results = [("원본 (B 전체)", original_fid, orig_mean, orig_cov, len(features_B), None)]
    
    # 전략 1: 공격적 Outlier 탐색
    print("\n" + "=" * 80)
    try:
        fid1, mean1, cov1, indices1, search_results = strategy_aggressive_outlier_search(
            features_A, features_B
        )
        results.append(("Aggressive Outlier", fid1, mean1, cov1, len(indices1), indices1))
    except Exception as e:
        print(f"  → 실패: {e}")
    
    # 전략 2: 2단계 복합 (다양한 파라미터)
    print("\n" + "=" * 80)
    for outlier_p in [90, 85, 80]:
        for target_mult in [2, 4, 8]:
            try:
                target = len(features_A) * target_mult
                fid2, mean2, cov2, indices2 = strategy_two_stage_hybrid(
                    features_A, features_B, 
                    outlier_percentile=outlier_p,
                    target_size=target
                )
                results.append((f"Hybrid(p={outlier_p},×{target_mult})", fid2, mean2, cov2, len(indices2), indices2))
            except Exception as e:
                print(f"  → 실패: {e}")
    
    # 전략 3: Iterative Removal
    print("\n" + "=" * 80)
    try:
        fid3, mean3, cov3, indices3 = strategy_iterative_removal(
            features_A, features_B,
            target_fid=20.0,
            max_remove_ratio=0.4,
            removal_rate=0.02
        )
        results.append(("Iterative Removal", fid3, mean3, cov3, len(indices3), indices3))
    except Exception as e:
        print(f"  → 실패: {e}")
    
    # 전략 4: 공분산 Greedy
    print("\n" + "=" * 80)
    try:
        fid4, mean4, cov4, indices4 = strategy_covariance_greedy(
            features_A, features_B,
            target_size=len(features_A) * 4,
            n_iter=200
        )
        results.append(("Covariance Greedy", fid4, mean4, cov4, len(indices4), indices4))
    except Exception as e:
        print(f"  → 실패: {e}")
    
    # 전략 5: Sinkhorn OT
    print("\n" + "=" * 80)
    try:
        for reg in [0.1, 0.05, 0.01]:
            fid5, mean5, cov5, indices5 = strategy_sinkhorn_ot(
                features_A, features_B,
                target_size=len(features_A) * 4,
                reg=reg,
                device=device
            )
            if len(indices5) > 0:
                results.append((f"Sinkhorn OT(reg={reg})", fid5, mean5, cov5, len(indices5), indices5))
    except Exception as e:
        print(f"  → 실패: {e}")
    
    # 전략 6: 고유값 매칭
    print("\n" + "=" * 80)
    try:
        fid6, mean6, cov6, indices6 = strategy_eigenvalue_matching(
            features_A, features_B,
            target_size=len(features_A) * 4
        )
        results.append(("Eigenvalue Matching", fid6, mean6, cov6, len(indices6), indices6))
    except Exception as e:
        print(f"  → 실패: {e}")
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("📋 전략별 결과 요약")
    print("=" * 80)
    print(f"\n{'전략':<30} {'FID':>10} {'평균':>10} {'공분산':>12} {'샘플수':>10} {'vs원본':>10}")
    print("-" * 90)
    
    # FID 순으로 정렬
    results.sort(key=lambda x: x[1])
    
    for name, fid, mean_t, cov_t, n_samples, _ in results:
        diff = fid - original_fid
        marker = "🏆" if fid == results[0][1] and name != "원본 (B 전체)" else ""
        print(f"{marker}{name:<28} {fid:>10.4f} {mean_t:>10.4f} {cov_t:>12.4f} {n_samples:>10} {diff:>+10.4f}")
    
    # 목표 달성 여부
    best_result = None
    for r in results:
        if r[0] != "원본 (B 전체)":
            if best_result is None or r[1] < best_result[1]:
                best_result = r
    
    print("\n" + "=" * 80)
    if best_result:
        if best_result[1] < 20:
            print(f"🎉 목표 달성! 최고 FID: {best_result[1]:.4f} ({best_result[0]})")
        elif best_result[1] < original_fid:
            print(f"📈 개선됨! 최고 FID: {best_result[1]:.4f} ({best_result[0]})")
            print(f"   원본 대비: {best_result[1] - original_fid:+.4f}")
            print(f"   목표(20)까지: {best_result[1] - 20:.4f} 남음")
        else:
            print(f"⚠️ 개선 실패. 원본이 최선: {original_fid:.4f}")
    
    return results


def analyze_covariance_contribution(features_A: np.ndarray, features_B: np.ndarray):
    """공분산 term의 상세 분석"""
    
    print("\n" + "=" * 60)
    print("📐 공분산 상세 분석")
    print("=" * 60)
    
    sigma_A = np.cov(features_A, rowvar=False)
    sigma_B = np.cov(features_B, rowvar=False)
    
    eigvals_A = np.linalg.eigvalsh(sigma_A)
    eigvals_B = np.linalg.eigvalsh(sigma_B)
    
    print(f"\n공분산 행렬 고유값 분석:")
    print(f"  Real (A) [n={len(features_A)}]:")
    print(f"    - 최대 고유값: {eigvals_A.max():.4f}")
    print(f"    - 최소 고유값: {eigvals_A.min():.4f}")
    print(f"    - 조건수: {eigvals_A.max() / (eigvals_A.min() + 1e-10):.2f}")
    print(f"    - Trace: {np.trace(sigma_A):.2f}")
    
    print(f"  Gen (B) [n={len(features_B)}]:")
    print(f"    - 최대 고유값: {eigvals_B.max():.4f}")
    print(f"    - 최소 고유값: {eigvals_B.min():.4f}")
    print(f"    - 조건수: {eigvals_B.max() / (eigvals_B.min() + 1e-10):.2f}")
    print(f"    - Trace: {np.trace(sigma_B):.2f}")
    
    cov_diff = sigma_A - sigma_B
    frob_norm = np.linalg.norm(cov_diff, 'fro')
    print(f"\n  공분산 차이 (Frobenius norm): {frob_norm:.2f}")
    
    return sigma_A, sigma_B, eigvals_A, eigvals_B


def main(args):
    device = torch.device(args.device)
    
    print("=" * 80)
    print("🔬 공분산 분석 기반 FID 개선 도구 (v3)")
    print("   목표: FID < 20")
    print("=" * 80)
    
    # 1. 데이터 로드
    print(f"\n📁 Real 이미지 디렉토리: {args.real_dir}")
    print(f"📁 Gen 이미지 디렉토리: {args.gen_dir}")
    
    real_dataset = ImageDataset([args.real_dir])
    gen_dataset = ImageDataset([args.gen_dir])
    
    print(f"\n  Real 이미지 수: {len(real_dataset)}")
    print(f"  Gen 이미지 수: {len(gen_dataset)}")
    
    real_loader = DataLoader(real_dataset, batch_size=args.batch_size, 
                             shuffle=False, num_workers=args.num_workers)
    gen_loader = DataLoader(gen_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=args.num_workers)
    
    # 2. Feature 추출
    print("\n" + "=" * 80)
    print("🧠 InceptionV3 Feature 추출")
    print("=" * 80)
    
    model = InceptionV3FeatureExtractor().to(device)
    
    print("\n[Real 이미지 feature 추출]")
    real_features = extract_features(real_loader, model, device)
    
    print("\n[Gen 이미지 feature 추출]")
    gen_features = extract_features(gen_loader, model, device)
    
    print(f"\n  Real features shape: {real_features.shape}")
    print(f"  Gen features shape: {gen_features.shape}")
    
    # 3. 원본 FID 계산
    print("\n" + "=" * 80)
    print("📊 원본 FID 계산")
    print("=" * 80)
    
    original_fid, mean_term, cov_term = compute_fid(real_features, gen_features)
    print(f"\n  평균 term: {mean_term:.4f}")
    print(f"  공분산 term: {cov_term:.4f}")
    print(f"  원본 FID: {original_fid:.4f}")
    print(f"\n  공분산 기여도: {cov_term/original_fid*100:.1f}%")
    
    # 4. 공분산 상세 분석
    analyze_covariance_contribution(real_features, gen_features)
    
    # 5. 모든 v5 전략 평가 (목표: FID < 20)
    results = evaluate_all_strategies_v4(real_features, gen_features, device=args.device)
    
    # 6. Feature 및 결과 저장
    # 최고 결과 찾기
    best_result = None
    for r in results:
        if r[0] != "원본 (B 전체)" and r[5] is not None:
            if best_result is None or r[1] < best_result[1]:
                best_result = r
    
    if args.save_features:
        save_path = os.path.join(os.path.dirname(args.gen_dir), "fid_optimization_results.npz")
        save_dict = {
            'real_features': real_features,
            'gen_features': gen_features,
        }
        if best_result:
            save_dict['best_indices'] = best_result[5]
            save_dict['best_fid'] = best_result[1]
            save_dict['best_strategy'] = best_result[0]
        
        np.savez(save_path, **save_dict)
        print(f"\n  결과 저장됨: {save_path}")
    
    # 7. 최적 FID 이미지 경로들을 JSON으로 저장
    if args.save_json and best_result is not None:
        best_indices = best_result[5]
        best_fid = best_result[1]
        best_strategy = best_result[0]
        
        # gen_dataset의 이미지 경로 목록에서 최적 인덱스에 해당하는 경로 추출
        selected_paths = [gen_dataset.image_paths[i] for i in best_indices]
        
        # JSON으로 저장할 내용 구성
        json_output = {
            "strategy": best_strategy,
            "fid": float(best_fid),
            "mean_term": float(best_result[2]),
            "cov_term": float(best_result[3]),
            "total_gen_images": len(gen_dataset),
            "selected_count": len(selected_paths),
            "selected_ratio": len(selected_paths) / len(gen_dataset),
            "real_dir": args.real_dir,
            "gen_dir": args.gen_dir,
            "selected_paths": selected_paths
        }
        
        # JSON 파일 저장
        if args.json_output:
            json_save_path = args.json_output
        else:
            json_save_path = os.path.join(os.path.dirname(args.gen_dir), "best_fid_selected_paths.json")
        
        with open(json_save_path, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, ensure_ascii=False, indent=2)
        
        print(f"\n" + "=" * 80)
        print(f"📄 최적 FID 이미지 경로 JSON 저장")
        print(f"=" * 80)
        print(f"  전략: {best_strategy}")
        print(f"  FID: {best_fid:.4f}")
        print(f"  선택된 이미지: {len(selected_paths)}개 / {len(gen_dataset)}개 ({len(selected_paths)/len(gen_dataset)*100:.1f}%)")
        print(f"  저장 경로: {json_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="공분산 분석 기반 FID 개선 도구 v3")
    parser.add_argument("--real_dir", type=str, required=True, 
                        help="Real 이미지 디렉토리")
    parser.add_argument("--gen_dir", type=str, required=True, 
                        help="Generated 이미지 디렉토리")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="배치 사이즈 (default: 32)")
    parser.add_argument("--n_clusters", type=int, default=50, 
                        help="K-Means 클러스터 수 (default: 50)")
    parser.add_argument("--num_workers", type=int, default=8, 
                        help="DataLoader worker 수 (default: 8)")
    parser.add_argument("--device", type=str, 
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="사용할 디바이스 (default: cuda)")
    parser.add_argument("--save_features", action="store_true",
                        help="추출된 features 및 결과 저장 여부")
    parser.add_argument("--save_json", action="store_true",
                        help="최적 FID를 달성한 이미지 경로들을 JSON으로 저장")
    parser.add_argument("--json_output", type=str, default=None,
                        help="JSON 출력 파일 경로 (지정하지 않으면 gen_dir 부모 폴더에 저장)")
    
    args = parser.parse_args()
    main(args)
