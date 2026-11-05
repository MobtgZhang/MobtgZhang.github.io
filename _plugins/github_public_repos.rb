# frozen_string_literal: true

require "json"
require "net/http"
require "uri"

module Jekyll
  # 抓取 GitHub 公开仓库的实时数据，供开源页展示。
  #
  # 站点构建时调用一次；页面上的 star / fork / 语言等数字随后由
  # assets/js/open-source.js 在浏览器端再次拉取并刷新，
  # 因此即使构建期被限流，访客看到的仍是实时数据。
  class GitHubPublicReposGenerator < Generator
    safe true
    priority :low

    def generate(site)
      site.data["github_public_repos"] = fetch_repos(site) || []
    end

    private

    def fetch_repos(site)
      user = site.config.dig("github", "username")
      return nil if user.nil? || user.to_s.empty?

      uri = URI("https://api.github.com/users/#{user}/repos?per_page=100&sort=updated")
      http = Net::HTTP.new(uri.host, uri.port)
      http.use_ssl = true
      http.open_timeout = 8
      http.read_timeout = 12

      req = Net::HTTP::Get.new(uri.request_uri)
      req["Accept"] = "application/vnd.github+json"
      req["User-Agent"] = "Jekyll-GitHubRepos/1.0"
      token = ENV["JEKYLL_GITHUB_TOKEN"] || ENV["GITHUB_TOKEN"] || ENV["GH_TOKEN"]
      req["Authorization"] = "Bearer #{token}" if token && !token.empty?

      res = http.request(req)
      unless res.is_a?(Net::HTTPSuccess)
        Jekyll.logger.warn "GitHubPublicRepos: API #{res.code} — 开源页将使用静态兜底数据，可设置 JEKYLL_GITHUB_TOKEN 后重试"
        return nil
      end

      list = JSON.parse(res.body)
      list.map { |r| normalize(r) }
    rescue JSON::ParserError, SocketError, IOError, Net::OpenTimeout, Net::ReadTimeout => e
      Jekyll.logger.warn "GitHubPublicRepos: #{e.class} — #{e.message}"
      nil
    end

    # 只保留页面真正用得到的字段，避免把整个 API 响应塞进 site.data。
    def normalize(r)
      {
        "name"        => r["name"],
        "html_url"    => r["html_url"],
        "description" => r["description"],
        "updated_at"  => r["updated_at"],
        "pushed_at"   => r["pushed_at"],
        "created_at"  => r["created_at"],
        "stars"       => r["stargazers_count"].to_i,
        "forks"       => r["forks_count"].to_i,
        "watchers"    => r["subscribers_count"].to_i,
        "open_issues" => r["open_issues_count"].to_i,
        "language"    => r["language"],
        "topics"      => Array(r["topics"]),
        "license"     => r["license"] && r["license"]["spdx_id"],
        "fork"        => r["fork"] == true,
        "archived"    => r["archived"] == true,
        "homepage"    => r["homepage"],
        "size"        => r["size"].to_i,
        "default_branch" => r["default_branch"]
      }
    end
  end
end
