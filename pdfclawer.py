import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, quote
import time
import re
from tqdm import tqdm
import argparse

class BingPDFCrawler:
    def __init__(self, download_folder="./bing_pdfs"):
        """
        初始化Bing PDF爬虫
        
        Args:
            download_folder: 下载文件夹路径
        """
        self.download_folder = download_folder
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        })
        
        # 创建下载文件夹
        if not os.path.exists(download_folder):
            os.makedirs(download_folder)
            
    def search_bing(self, query, num_results=20, max_pages=5):
        """
        在Bing上搜索并获取结果URL列表
        
        Args:
            query: 搜索关键词
            num_results: 要获取的结果数量
            max_pages: 最大搜索页数
            
        Returns:
            list: 搜索结果URL列表
        """
        all_pdf_links = []
        
        # 清理查询字符串，移除可能导致编码问题的字符
        query = self.clean_query(query)
        
        # 多种搜索方式
        search_queries = [
            query + " filetype:pdf",  # 标准PDF搜索
            query,                    # 不加限制，后续过滤
            query + " pdf",          # 添加pdf关键词
        ]
        
        for search_query in search_queries:
            if len(all_pdf_links) >= num_results * 2:
                break
            
            # 安全编码查询字符串
            try:
                encoded_query = quote(search_query, safe='')
            except (UnicodeEncodeError, UnicodeError):
                # 如果编码失败，尝试先编码为utf-8
                try:
                    encoded_query = quote(search_query.encode('utf-8'), safe='')
                except:
                    print(f"警告: 无法编码查询词 '{search_query}'，跳过此搜索")
                    continue
            
            for page in range(1, max_pages + 1):
                if len(all_pdf_links) >= num_results * 2:
                    break
                    
                # Bing的URL格式
                search_url = f"https://www.bing.com/search?q={encoded_query}&first={(page-1)*10 + 1}"
                
                try:
                    print(f"正在搜索第{page}页: {search_query}")
                    response = self.session.get(search_url, timeout=10)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # 查找所有搜索结果链接
                    links_found = []
                    
                    # 方法1：查找所有a标签
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        if href.startswith('http') and 'bing.com' not in href:
                            links_found.append(href)
                    
                    # 去重
                    links_found = list(set(links_found))
                    
                    for href in links_found:
                        # 检查是否为PDF
                        if '.pdf' in href.lower():
                            if href not in all_pdf_links:
                                all_pdf_links.append(href)
                                print(f"  找到PDF: {href[:80]}...")
                    
                    # 页面延迟
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"搜索第{page}页时出错: {e}")
                    continue
        
        # 去重
        all_pdf_links = list(set(all_pdf_links))
        print(f"\n总共找到 {len(all_pdf_links)} 个PDF链接")
        
        return all_pdf_links[:num_results * 2]
    
    def clean_query(self, query):
        """清理查询字符串，移除可能导致问题的字符"""
        # 移除emoji和特殊符号
        # 保留中文、英文、数字、空格、常见标点
        cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s\+\-\(\)]', '', query)
        # 确保不是空字符串
        if not cleaned.strip():
            cleaned = "document"
        return cleaned.strip()
    
    def download_pdf(self, url, filename=None):
        """
        下载单个PDF文件
        """
        try:
            # 获取文件名
            if filename is None:
                filename = self.extract_filename_from_url(url)
            
            # 清理文件名
            filename = self.sanitize_filename(filename)
            if not filename.endswith('.pdf'):
                filename += '.pdf'
                
            filepath = os.path.join(self.download_folder, filename)
            
            # 如果文件已存在，跳过
            if os.path.exists(filepath):
                print(f"文件已存在，跳过: {filename}")
                return False
            
            # 下载文件
            print(f"正在下载: {filename}")
            response = self.session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # 检查内容类型
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
                print(f"警告: {filename} 可能不是PDF文件 (Content-Type: {content_type})")
            
            # 获取文件大小
            total_size = int(response.headers.get('content-length', 0))
            
            # 使用tqdm显示下载进度
            with open(filepath, 'wb') as file:
                if total_size == 0:
                    file.write(response.content)
                else:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                        for data in response.iter_content(chunk_size=1024):
                            file.write(data)
                            pbar.update(len(data))
            
            print(f"✓ 下载完成: {filename}")
            return True
            
        except Exception as e:
            print(f"✗ 下载失败: {e}")
            return False
    
    def extract_filename_from_url(self, url):
        """从URL中提取有意义的文件名"""
        try:
            parsed = urlparse(url)
            path = parsed.path
            
            # 尝试从URL获取文件名
            filename = os.path.basename(path)
            
            # 如果文件名无效或太简单，从URL路径提取关键词
            if not filename or len(filename) < 5 or '.' not in filename:
                # 提取路径中的关键词
                path_parts = path.split('/')
                for part in reversed(path_parts):
                    if part and len(part) > 5 and not part.isdigit():
                        filename = part
                        break
                
                # 如果还是无效，使用时间戳和URL的hash
                if not filename:
                    filename = f"pdf_{int(time.time())}_{abs(hash(url)) % 10000}"
            
            return filename
        except Exception:
            # 如果出错，返回时间戳文件名
            return f"pdf_{int(time.time())}"
    
    def sanitize_filename(self, filename):
        """清理文件名，保留中文"""
        try:
            # 只替换Windows非法字符，保留中文
            illegal_chars = r'[<>:"/\\|?*]'
            filename = re.sub(illegal_chars, '_', filename)
            # 移除控制字符
            filename = re.sub(r'[\x00-\x1f\x7f]', '', filename)
            # 移除可能导致问题的Unicode字符
            filename = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9_.\-]', '_', filename)
            # 限制长度
            if len(filename) > 150:
                name, ext = os.path.splitext(filename)
                filename = name[:146] + ext
            return filename.strip()
        except Exception:
            # 如果出错，返回简单的时间戳
            return f"pdf_{int(time.time())}"
    
    def crawl_and_download(self, query, max_results=20, delay=1):
        """
        爬取并下载PDF文件
        """
        # 搜索获取PDF链接
        pdf_urls = self.search_bing(query, max_results * 2, max_pages=3)
        
        if not pdf_urls:
            print("❌ 未找到PDF文件")
            print("建议：")
            print("  1. 尝试更宽泛的关键词，如：脂溢性皮炎 指南")
            print("  2. 尝试英文关键词：seborrheic dermatitis")
            print("  3. 减少搜索精度要求")
            print("  4. 尝试单个关键词而不是多个")
            return
        
        # 显示找到的PDF列表
        print(f"\n找到的PDF文件列表:")
        for i, url in enumerate(pdf_urls[:max_results], 1):
            filename = self.extract_filename_from_url(url)
            print(f"  {i}. {filename[:60]}...")
        
        print(f"\n开始下载 {min(len(pdf_urls), max_results)} 个PDF文件...")
        print(f"保存位置: {os.path.abspath(self.download_folder)}")
        print("-" * 60)
        
        success_count = 0
        for i, url in enumerate(pdf_urls[:max_results]):
            print(f"\n[{i+1}/{min(len(pdf_urls), max_results)}]")
            if self.download_pdf(url):
                success_count += 1
            
            # 下载延迟
            if i < len(pdf_urls[:max_results]) - 1:
                time.sleep(delay)
        
        print("\n" + "=" * 60)
        print(f"✅ 下载完成！成功: {success_count}, 失败: {min(len(pdf_urls), max_results) - success_count}")
        print(f"📁 文件保存在: {os.path.abspath(self.download_folder)}")

def main():
    parser = argparse.ArgumentParser(description='Bing PDF爬虫工具 - 专业版')
    parser.add_argument('query', nargs='?', default='python tutorial', 
                       help='搜索关键词')
    parser.add_argument('-n', '--number', type=int, default=10,
                       help='要下载的PDF数量')
    parser.add_argument('-o', '--output', default='./bing_pdfs',
                       help='下载文件夹路径')
    parser.add_argument('-d', '--delay', type=float, default=1.0,
                       help='请求延迟秒数')
    
    args = parser.parse_args()
    
    crawler = BingPDFCrawler(download_folder=args.output)
    crawler.crawl_and_download(
        query=args.query,
        max_results=args.number,
        delay=args.delay
    )

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        print("=" * 60)
        print("Bing PDF 爬虫工具 - 专业版")
        print("=" * 60)
        
        query = input("请输入搜索关键词 (默认: 脂溢性皮炎 指南): ").strip()
        if not query:
            query = "脂溢性皮炎 指南"
        
        # 清理用户输入的查询词
        crawler_temp = BingPDFCrawler()
        query = crawler_temp.clean_query(query)
        
        try:
            num = int(input("请输入要下载的PDF数量 (默认: 10): ") or "10")
        except ValueError:
            num = 10
        
        folder = input("请输入保存文件夹路径 (默认: ./bing_pdfs): ").strip()
        if not folder:
            folder = "./bing_pdfs"
        
        crawler = BingPDFCrawler(download_folder=folder)
        crawler.crawl_and_download(query, num, delay=2.0)
    else:
        main()