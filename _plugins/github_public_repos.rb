# frozen_string_literal: true

require "json"
require "net/http"
require "uri"

module Jekyll
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
        Jekyll.logger.warn "GitHubPublicRepos: API #{res.code} — 开源页列表将为空，可设置 JEKYLL_GITHUB_TOKEN 后重试"
        return nil
      end

      list = JSON.parse(res.body)
      list.map do |r|
        {
          "name" => r["name"],
          "html_url" => r["html_url"],
          "description" => r["description"],
          "updated_at" => r["updated_at"],
          "stargazers_count" => r["stargazers_count"].to_i,
          "fork" => r["fork"] == true
        }
      end
    rescue JSON::ParserError, SocketError, IOError, Net::OpenTimeout, Net::ReadTimeout => e
      Jekyll.logger.warn "GitHubPublicRepos: #{e.class} — #{e.message}"
      nil
    end
  end
end
