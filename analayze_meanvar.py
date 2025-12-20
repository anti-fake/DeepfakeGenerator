import argparse
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def list_paths_recursive(root_dirs: List[str]) -> List[str]:
    """디렉토리에서 이미지 경로를 재귀적으로 수집"""
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
        self.image_paths.sort()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((299, 299), antialias=True),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image


class InceptionV3FeatureExtractor(nn.Module):
    """Inception V3 네트워크에서 2048차원 feature 추출"""
    def __init__(self):
        super().__init__()
        from torchvision.models import inception_v3, Inception_V3_Weights
        inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        
        # Pool layer 전까지의 네트워크 사용
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
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = x.view(x.size(0), -1)  # (batch_size, 2048)
        return x


def extract_features(dataloader: DataLoader, model: nn.Module, device: str) -> np.ndarray:
    """데이터로더에서 Inception feature 추출"""
    features_list = []
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            batch = batch.to(device)
            features = model(batch)
            features_list.append(features.cpu().numpy())
    
    return np.concatenate(features_list, axis=0)


def diagnose_fid_bottleneck(features_A: np.ndarray, features_B: np.ndarray):
    """FID가 안 줄어드는 원인 파악 - 상세 분석"""
    
    mu_A = np.mean(features_A, axis=0)
    mu_B = np.mean(features_B, axis=0)
    sigma_A = np.cov(features_A, rowvar=False)
    sigma_B = np.cov(features_B, rowvar=False)
    
    print("=" * 60)
    print("📊 Feature 통계 분석")
    print("=" * 60)
    
    # 기본 통계
    print(f"\n[Real Data]")
    print(f"  - 샘플 수: {features_A.shape[0]}")
    print(f"  - Feature 차원: {features_A.shape[1]}")
    print(f"  - 평균의 평균: {np.mean(mu_A):.6f}")
    print(f"  - 평균의 표준편차: {np.std(mu_A):.6f}")
    print(f"  - 평균의 최소/최대: [{np.min(mu_A):.6f}, {np.max(mu_A):.6f}]")
    print(f"  - 공분산 대각 평균: {np.mean(np.diag(sigma_A)):.6f}")
    print(f"  - 공분산 Frobenius norm: {np.linalg.norm(sigma_A, 'fro'):.6f}")
    
    print(f"\n[Generated Data]")
    print(f"  - 샘플 수: {features_B.shape[0]}")
    print(f"  - Feature 차원: {features_B.shape[1]}")
    print(f"  - 평균의 평균: {np.mean(mu_B):.6f}")
    print(f"  - 평균의 표준편차: {np.std(mu_B):.6f}")
    print(f"  - 평균의 최소/최대: [{np.min(mu_B):.6f}, {np.max(mu_B):.6f}]")
    print(f"  - 공분산 대각 평균: {np.mean(np.diag(sigma_B)):.6f}")
    print(f"  - 공분산 Frobenius norm: {np.linalg.norm(sigma_B, 'fro'):.6f}")
    
    # 평균 차이 분석
    mean_diff = mu_A - mu_B
    print(f"\n[평균 차이 분석]")
    print(f"  - 평균 차이의 L2 norm: {np.linalg.norm(mean_diff):.6f}")
    print(f"  - 평균 차이의 평균: {np.mean(mean_diff):.6f}")
    print(f"  - 평균 차이의 표준편차: {np.std(mean_diff):.6f}")
    print(f"  - 평균 차이의 최소/최대: [{np.min(mean_diff):.6f}, {np.max(mean_diff):.6f}]")
    
    # 가장 큰 차이를 보이는 차원들
    top_k = 10
    top_diff_indices = np.argsort(np.abs(mean_diff))[-top_k:][::-1]
    print(f"  - 상위 {top_k}개 차이 차원: {top_diff_indices}")
    print(f"  - 상위 {top_k}개 차이 값: {mean_diff[top_diff_indices]}")
    
    # 공분산 차이 분석
    cov_diff = sigma_A - sigma_B
    print(f"\n[공분산 차이 분석]")
    print(f"  - 공분산 차이 Frobenius norm: {np.linalg.norm(cov_diff, 'fro'):.6f}")
    print(f"  - 대각 요소 차이의 평균: {np.mean(np.diag(cov_diff)):.6f}")
    print(f"  - 대각 요소 차이의 표준편차: {np.std(np.diag(cov_diff)):.6f}")
    
    # Eigenvalue 분석
    eigvals_A = np.linalg.eigvalsh(sigma_A)
    eigvals_B = np.linalg.eigvalsh(sigma_B)
    print(f"\n[Eigenvalue 분석]")
    print(f"  - Real 상위 10개 eigenvalues: {eigvals_A[-10:][::-1]}")
    print(f"  - Gen 상위 10개 eigenvalues: {eigvals_B[-10:][::-1]}")
    print(f"  - Real eigenvalue 합: {np.sum(eigvals_A):.6f}")
    print(f"  - Gen eigenvalue 합: {np.sum(eigvals_B):.6f}")
    
    # FID 분해
    print("\n" + "=" * 60)
    print("📈 FID 분해 분석")
    print("=" * 60)
    
    mean_term = np.sum((mu_A - mu_B) ** 2)
    
    # sqrt(sigma_A @ sigma_B) 계산
    covmean = linalg.sqrtm(sigma_A @ sigma_B)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            print("  ⚠️ 경고: covmean에 유의미한 허수부가 있음")
        covmean = covmean.real
    
    cov_term = np.trace(sigma_A + sigma_B - 2 * covmean)
    
    total_fid = mean_term + cov_term
    
    print(f"\n  평균 차이 기여 (mean term): {mean_term:.4f} ({100*mean_term/total_fid:.1f}%)")
    print(f"  공분산 차이 기여 (cov term): {cov_term:.4f} ({100*cov_term/total_fid:.1f}%)")
    print(f"  ───────────────────────────────")
    print(f"  총 FID: {total_fid:.4f}")
    
    # 개선 방향 제안
    print("\n" + "=" * 60)
    print("💡 분석 결과 요약")
    print("=" * 60)
    
    if mean_term > cov_term:
        print(f"\n  → 평균 차이가 FID의 주요 원인 ({100*mean_term/total_fid:.1f}%)")
        print("  → 생성 이미지의 전반적인 색상, 밝기, 또는 스타일이 실제와 다름")
    else:
        print(f"\n  → 공분산 차이가 FID의 주요 원인 ({100*cov_term/total_fid:.1f}%)")
        print("  → 생성 이미지의 다양성 또는 feature 간 상관관계가 실제와 다름")
    
    return {
        'mean_term': mean_term,
        'cov_term': cov_term,
        'total_fid': total_fid,
        'mu_real': mu_A,
        'mu_gen': mu_B,
        'sigma_real': sigma_A,
        'sigma_gen': sigma_B,
    }


def main(args):
    print(f"Real 데이터 경로: {args.real_dir}")
    print(f"Generated 데이터 경로: {args.gen_dir}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    
    # 데이터셋 로드
    real_dataset = ImageDataset([args.real_dir])
    gen_dataset = ImageDataset([args.gen_dir])
    
    print(f"\nReal 이미지 수: {len(real_dataset)}")
    print(f"Generated 이미지 수: {len(gen_dataset)}")
    
    real_loader = DataLoader(real_dataset, batch_size=args.batch_size, 
                             shuffle=False, num_workers=args.num_workers)
    gen_loader = DataLoader(gen_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=args.num_workers)
    
    # Feature extractor 로드
    print("\nInception V3 모델 로딩...")
    model = InceptionV3FeatureExtractor()
    model.to(args.device)
    
    # Feature 추출
    print("\nReal 데이터 feature 추출 중...")
    features_real = extract_features(real_loader, model, args.device)
    
    print("\nGenerated 데이터 feature 추출 중...")
    features_gen = extract_features(gen_loader, model, args.device)
    
    print(f"\nReal features shape: {features_real.shape}")
    print(f"Generated features shape: {features_gen.shape}")
    
    # FID 분석
    results = diagnose_fid_bottleneck(features_real, features_gen)
    
    # 결과 저장 (옵션)
    if args.save_features:
        save_path = args.save_features
        np.savez(save_path, 
                 features_real=features_real, 
                 features_gen=features_gen,
                 mu_real=results['mu_real'],
                 mu_gen=results['mu_gen'],
                 sigma_real=results['sigma_real'],
                 sigma_gen=results['sigma_gen'])
        print(f"\n✅ Features 저장됨: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FID 분석: 평균과 공분산 비교")
    parser.add_argument("--real_dir", type=str, required=True, 
                        help="Real 이미지 디렉토리 경로")
    parser.add_argument("--gen_dir", type=str, required=True, 
                        help="Generated 이미지 디렉토리 경로")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size (default: 32)")
    parser.add_argument("--device", type=str, 
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (default: cuda if available)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader workers (default: 8)")
    parser.add_argument("--save_features", type=str, default=None,
                        help="Feature 저장 경로 (.npz)")
    
    args = parser.parse_args()
    main(args)
