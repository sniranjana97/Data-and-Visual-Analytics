import http.client
import json
import time
import sys
import collections
import csv
import os

counter = 0

movie_filename = "movie_ID_name.csv"
movie_similarity_filename = "movie_ID_sim_movie_ID.csv"
api_key = sys.argv[1];
conn = http.client.HTTPSConnection("api.themoviedb.org")
payload = {}

def get_api_response(request):
  api_key = sys.argv[1];

  conn = http.client.HTTPSConnection("api.themoviedb.org")
  payload = {}
  conn.request("GET", request, payload)

  res = conn.getresponse()
  data = res.read().decode("utf-8")
  json_object = json.loads(data)
  return json_object, res.getheaders()

def remove_file(filename):
  try:
    os.remove(filename)
  except OSError:
    pass

def is_not_duplicate(l, i):
  for index in range(len(l)):
    if (i != index and sorted(l[i]) == sorted(l[index]) and l[i] != sorted(l[i])):
      return False
  return True


def get_movie_list():
  remove_file(movie_filename)
  movie_data = open(movie_filename, 'a')
  csvwriter = csv.writer(movie_data)

  for i in range(18):
    request = "/3/discover/movie?with_genres=18&primary_release_date.gte=2004-01-01"\
    "&sort_by=popularity.desc&api_key=" + api_key + "&page=" + str(i+1)
    json_object, headers = get_api_response(request)

    if('results' not in json_object):
      time.sleep(int(headers[4][1]) + 1)
      json_object, headers = get_api_response(request)

    results = json_object['results']
    if i == 17:
      results = results[:10]
    for result in results:
      csvwriter.writerow([result['id'], result['title']])
  movie_data.close()

def get_similar_movies():
  remove_file(movie_similarity_filename)

  movie_data = open(movie_filename)
  csv_reader = csv.reader(movie_data)

  movie_similarity_data = open(movie_similarity_filename, 'a')
  csv_writer = csv.writer(movie_similarity_data)

  similar_movies = []
  for row in csv_reader:
    movie_id = row[0]
    request = "https://api.themoviedb.org/3/movie/" + str(movie_id) \
    + "/similar?api_key=" + str(api_key) + "&page=1"

    json_object, headers = get_api_response(request)

    if('results' not in json_object):
      time.sleep(int(headers[4][1]) + 1)
      json_object, headers = get_api_response(request)

    results = json_object['results']
    results = results[:5]
    for result in results:
      similar_movies.append([int(movie_id), int(result['id'])])
  #print(len(similar_movies))
  similar_movies_without_duplicates = [similar_movies[i] for i in range(len(similar_movies)) if is_not_duplicate(similar_movies, i)]

  remove_file(movie_similarity_filename)
  movie_similarity_data = open(movie_similarity_filename, 'a')
  csv_writer = csv.writer(movie_similarity_data)
  for similar_movie in similar_movies_without_duplicates:
      csv_writer.writerow(similar_movie)
  movie_data.close()
  movie_similarity_data.close()

def insert_header():

  movie_similarity_data_read = open(movie_similarity_filename)
  csv_reader = csv.reader(movie_similarity_data_read)
  data = [row for row in csv_reader]
  remove_file(movie_similarity_filename)

  movie_similarity_data_write = open(movie_similarity_filename, 'a')
  csv_writer = csv.writer(movie_similarity_data_write)
  csv_writer.writerow(['Source', 'Target'])
  csv_writer.writerows(data)

  movie_similarity_data_write.close()
  movie_similarity_data_read.close()

if __name__ == '__main__':
  get_movie_list()
  get_similar_movies()
  insert_header()