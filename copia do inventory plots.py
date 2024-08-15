import random, math,sys
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon,box
import shapely
from sklearn.cluster import KMeans
class PlotAlocation:

    def __init__(self, shp_dir, epsg=None):
        self.shp = gpd.read_file(shp_dir).to_crs(epsg=epsg)
        self.polygons = self.shp['geometry']
        self.epgs = epsg
        self.total_area = self.shp['geometry'].area.sum()

    def validate_sampling(self):
        if self.max_number_of_points * self.plot_area > self.reduced_shp_total_area:
            print("Error: The 'plot_area' * 'sample_number' exceeds the total area size. Try reducing 'plot_area' or 'sample_number'.")
            sys.exit()

    def create_plots(self, distribution='best sampling', plot_format='round',plot_area=400, rectangle_size=None,
                     sample_number=0.1, min_border_distance=0, x_y_angle=None, save_buffer=False,
                     show_plot=True, save_dir=None):
        available_distributions = ['random','best sampling', 'systematic', 'systematic custom']
        available_plot_formats = ['round','squared','rectangle']

        if distribution not in available_distributions:
            print(f"'{distribution}' not in available distributions.")
            print(f"Available distribution: {available_distributions}")
            sys.exit()

        if plot_format not in available_plot_formats:
            print(f"'{plot_format}' not in available plot formats.")
            print(f"Available plot formats: {available_plot_formats}")
            sys.exit()

        self.distribution = distribution
        self.plot_format = plot_format
        self.plot_area = plot_area
        self.min_border_distance = min_border_distance
        self.reduced_shp = self.get_reduced_polygons_to_min_border_distance(self.shp)
        self.reduced_shp_total_area =self.reduced_shp['geometry'].area.sum()
        self.sample_number = sample_number
        if self.sample_number < 1:
            self.max_number_of_points = math.ceil((self.sample_number * self.total_area) / self.plot_area)
        else:
            self.max_number_of_points = self.sample_number

        self.reduced_shp['area_prop'] = self.reduced_shp['geometry'].area / self.reduced_shp_total_area
        self.reduced_shp['parcelas'] = np.ceil(self.reduced_shp['area_prop'] * self.max_number_of_points).astype(int)

        # Corrige o total para garantir que a soma seja exatamente igual a self.max_number_of_points
        total_parcelas = self.reduced_shp['parcelas'].sum()
        diferenca = total_parcelas - self.max_number_of_points

        # Ajusta a diferença removendo parcelas dos polígonos com mais parcelas, se necessário
        while diferenca > 0:
            # Identifica os índices dos polígonos com o maior número de parcelas
            max_parcelas_index = \
            self.reduced_shp[self.reduced_shp['parcelas'] == self.reduced_shp['parcelas'].max()].index[0]
            # Reduz o número de parcelas do primeiro polígono com o maior número de parcelas
            self.reduced_shp.at[max_parcelas_index, 'parcelas'] -= 1
            diferenca -= 1

        self.save_buffer = save_buffer
        self.show_plot = show_plot
        self.save_dir = save_dir
        self.x_y_angle = x_y_angle
        self.rectangle_size = rectangle_size

        if self.plot_format == 'rectangle':
            print("Atention! 'plot_area' is being calculated based on 'rectangle_size'")
            self.plot_area = self.rectangle_size[0] * self.rectangle_size[1]
            if not (isinstance(self.rectangle_size, tuple) and len(self.rectangle_size) == 2):
                print(
                    "rectangle_size must be a tuple containing exactly two numeric values (width, height). Ex:(20,20)")
                sys.exit()
            if not (isinstance(self.rectangle_size[0], (int, float)) and isinstance(self.rectangle_size[1],(int, float))):
                print("Both elements in rectangle_size must be numeric values (int or float). Ex:(20,20)")
                sys.exit()

        if self.save_dir is None:
            self.show_plot = True



        self.validate_sampling()

        if self.distribution == "random":
            points = self._generate_random_sample_points()
        if self.distribution == "best sampling":
            points = self._generate_best_sample_points()
        if self.distribution == "systematic":
            points = self._generate_systematic_sample_points()
        if self.distribution == "systematic custom":
            points = self._generate_systematic_custom_sample_points()
        buffer = self._create_buffer(points)

        if self.save_dir is not None:
            if not self.save_dir.endswith('.shp'):
                print("Error: save_dir must end with .shp")
                sys.exit()
            else:
                self._save_sample_points_to_shp(points, self.save_dir)
                if self.save_buffer == True:
                    self._save_sample_buffer_to_shp(buffer,"buffer.shp")

        if self.show_plot == True:
            self._visualize_points(points,buffer)

    def _poligon_contains_point(self, point):
        for index, row in self.reduced_shp.iterrows():
            polygon = row['geometry']
            if polygon.contains(point):
                return True

    def get_reduced_polygons_to_min_border_distance(self, shp):
        """
        Essa função cria uma versão dos poligonos que seja menor min_border_distance do que os poligonos reais,
        isso faz com que os poligonos em que as parcelas serão alocadas já estejam considerando a distancia de borda.
        Se o polígono reduzido ficar negativo ou muito pequeno, ele será excluído.
        """
        reduced_polygons = []

        for poly in shp.geometry:
            reduced_poly = poly.buffer(-self.min_border_distance)
            if reduced_poly.is_valid and not reduced_poly.is_empty and reduced_poly.area > 0:
                reduced_polygons.append(reduced_poly)

        reduced_polygons_gdf = gpd.GeoDataFrame(geometry=reduced_polygons, crs=shp.crs)
        return reduced_polygons_gdf

    def create_all_possible_points(self, with_index_flag=False):
        """Essa função cria todos os pontos possíveis, de acordo com grid_spacing
        se with_index_flag==True, retorna uma lista de tuplas contendo o número do poligono e as cordenadas do ponto.
        se with_index_flag==False, retorna apenas a lista de pontos."""
        extent = self.reduced_shp.total_bounds

        x_min, y_min, x_max, y_max = extent

        if not (np.isfinite(x_min) and np.isfinite(y_min) and np.isfinite(x_max) and np.isfinite(y_max)):
            print("Extensão do layer contém valores infinitos ou NaN.")
            return None

        if x_min >= x_max or y_min >= y_max:
            print("Intervalo de valores de x ou y inválido.")
            return None

        # Obtem a distancia que uma parcela deve ter da outra
        grid_spacing = self.get_distances()[1]
        valid_points = []

        x_coords = np.arange(x_min + grid_spacing / 2, x_max, grid_spacing)
        y_coords = np.arange(y_min + grid_spacing / 2, y_max, grid_spacing)

        if with_index_flag:
            for x in x_coords:
                for y in y_coords:
                    point = Point(x, y)
                    for index, poly in enumerate(self.reduced_shp.geometry):
                        if poly.contains(point):
                            valid_points.append((index, point))  # Salva o índice do polígono junto com o ponto
                            break
        else:
            for x in x_coords:
                for y in y_coords:
                    point = Point(x, y)
                    for poly in self.reduced_shp.geometry:
                        if poly.contains(point):
                            valid_points.append(point)
            return valid_points
        return valid_points

    def get_distances(self):
        """obtem a distancia que os pontos devem manter da borda e também a distancia que um ponto deve ter de outro"""
        if self.plot_format == 'round':
            add_value = math.sqrt(self.plot_area / math.pi)
        if self.plot_format == 'squared':
            add_value = (math.sqrt(self.plot_area) * math.sqrt(2)) / 2
        if self.plot_format == 'rectangle':
            add_value = math.sqrt((self.rectangle_size[0]/2) ** 2 + (self.rectangle_size[1]/2) ** 2)
        distance_from_each_other = add_value
        distance_from_border = self.min_border_distance + add_value
        return distance_from_border, distance_from_each_other
    def _validate_points(self, point, existing_points):
        """função que faz a validação dos pontos criados"""
        #Verifica se o ponto está contido em algum dos poligonos de self.shp

        if self.plot_format == 'round':
            radius = math.sqrt(self.plot_area / math.pi)

            for index, row in self.shp.iterrows():
                polygon = row['geometry']
                if isinstance(polygon, Polygon):
                    # Calcular a distância entre o ponto e a borda do polígono
                    distance_to_border = point.distance(polygon.exterior)

                    # Verificar se a distância está dentro do limite especificado
                    if distance_to_border <= self.get_distances()[0]:
                        return False

            # Verifica a sobreposição com pontos existentes
            for existing_point in existing_points:
                if point.distance(existing_point) < (radius*2):
                    return False

            return True

        if self.plot_format == 'squared':

            half_side = math.sqrt(self.plot_area) / 2
            min_distance = self.get_distances()[0]

            point_x, point_y = point.x, point.y

            # Define the square boundaries
            min_x, min_y = point_x - half_side, point_y - half_side
            max_x, max_y = point_x + half_side, point_y + half_side

            # Check for overlap with existing squares
            for existing_point in existing_points:
                existing_x, existing_y = existing_point.x, existing_point.y
                existing_min_x, existing_min_y = existing_x - half_side, existing_y - half_side
                existing_max_x, existing_max_y = existing_x + half_side, existing_y + half_side

                # Check if the squares overlap
                if not (
                        min_x >= existing_max_x or max_x <= existing_min_x or min_y >= existing_max_y or max_y <= existing_min_y):
                    return False

            # Check the distance between the square vertices and the polygon boundary
            square_vertices = [
                Point(min_x, min_y),
                Point(min_x, max_y),
                Point(max_x, min_y),
                Point(max_x, max_y)
            ]

            for vertex in square_vertices:
                for index, row in self.shp.iterrows():
                    polygon = row['geometry']
                    if isinstance(polygon, Polygon):
                        distance_to_boundary = vertex.distance(polygon.exterior)
                        if distance_to_boundary < min_distance:
                            return False

            return True

        if self.plot_format == 'rectangle':
            result = self._check_rectangle(point,existing_points,self.rectangle_size[0],self.rectangle_size[1])
            return result

        return False

    def _check_rectangle(self, point, existing_points, width, height):
        """
        Verifica se um retângulo centrado em 'point' é válido.

        :param point: Ponto centrado no retângulo.
        :param existing_points: Lista de pontos dos centros dos retângulos existentes.
        :param width: Largura do retângulo.
        :param height: Altura do retângulo.
        :return: True se o retângulo estiver completamente contido em uma feature, não intersectar nenhum retângulo existente, e estiver a pelo menos self.min_border_distance dos limites dos polígonos. False caso contrário.
        """
        half_width = width / 2
        half_height = height / 2

        # Define os cantos do retângulo ao redor do ponto central
        rectangle_points = [
            (point.x - half_width, point.y - half_height),
            (point.x + half_width, point.y - half_height),
            (point.x + half_width, point.y + half_height),
            (point.x - half_width, point.y + half_height),
            (point.x - half_width, point.y - half_height)  # Fecha o polígono
        ]

        # Cria a geometria do retângulo como um polígono
        rectangle_geom = Polygon(rectangle_points)

        # Verifica se o retângulo está completamente dentro de pelo menos uma feature na camada
        for feature in self.shp.geometry:
            feature_geom = feature

            # Buffer negativo da feature para garantir distância mínima dos limites
            buffered_feature_geom = feature_geom.buffer(-self.min_border_distance)

            if all(buffered_feature_geom.contains(Point(pt)) for pt in rectangle_points):
                # Verifica se o retângulo não intersecta nenhum retângulo dos existing_points
                for existing_point in existing_points:
                    existing_rectangle_points = [
                        (existing_point.x - half_width, existing_point.y - half_height),
                        (existing_point.x + half_width, existing_point.y - half_height),
                        (existing_point.x + half_width, existing_point.y + half_height),
                        (existing_point.x - half_width, existing_point.y + half_height),
                        (existing_point.x - half_width, existing_point.y - half_height)  # Fecha o polígono
                    ]
                    existing_rectangle_geom = Polygon(existing_rectangle_points)
                    if rectangle_geom.intersects(existing_rectangle_geom):
                        return False  # Se intersecta com um retângulo existente, é inválido

                return True  # Se está contido em uma feature e não intersecta retângulos existentes, é válido

        return False
    # =================================================
    """Distribuição randomica"""

    def _generate_random_sample_points(self, max_attempts=3000):

        extent = self.reduced_shp.total_bounds
        x_min, y_min, x_max, y_max = extent[0], extent[1], extent[2], extent[3]

        if not (np.isfinite(x_min) and np.isfinite(y_min) and np.isfinite(x_max) and np.isfinite(y_max)):
            print("Extensão do layer contém valores infinitos ou NaN.")
            return None

        if x_min >= x_max or y_min >= y_max:
            print("Intervalo de valores de x ou y inválido.")
            return None

        valid_points = []
        attempts = 0

        while len(valid_points) < self.max_number_of_points and attempts < max_attempts:
            try:
                x = np.random.uniform(x_min, x_max)
                y = np.random.uniform(y_min, y_max)
            except OverflowError as e:
                print(f"Erro ao gerar valores x ou y: {e}")
                break

            point = Point(x, y)

            if point is not None and self._poligon_contains_point(point) and self._validate_points(point, valid_points):
                valid_points.append(point)

            attempts += 1

        if len(valid_points) < self.max_number_of_points:
            print(
                "Warning! Unable to generate plots with the established criteria. Only the possible plots were generated.")

        if valid_points:
            points_gdf = gpd.GeoDataFrame(geometry=valid_points, crs=self.shp.crs)
            return points_gdf
        else:
            print("Não foi possível gerar parcelas com os critérios estabelecidos.")
            return None
    """ FIM Distribuição randomica"""
    # =================================================
    """Distribuição Best Sampling"""
    def _generate_best_sample_points(self):
        print("Searching for the best possible allocation, this may take a few seconds...")
        print(self.reduced_shp['parcelas'])
        valid_points = self.create_all_possible_points(with_index_flag=False)


        points_gdf = gpd.GeoDataFrame(geometry=valid_points, crs=self.shp.crs)

        if len(points_gdf) < self.max_number_of_points:
            print("Unable to generate plots with the established criteria. Only the possible plots were generated.")
            return points_gdf

        final_points = []

        for idx, row in self.reduced_shp.iterrows():
            polygon = row['geometry']
            num_parcelas = row['parcelas']

            points_within_polygon = points_gdf[points_gdf.within(polygon)]

            if len(points_within_polygon) <= num_parcelas:
                final_points.append(points_within_polygon)
                continue

            # Use k-means to find the best cluster centers
            kmeans = KMeans(n_clusters=num_parcelas, random_state=0).fit(
                np.array(list(points_within_polygon.geometry.apply(lambda point: (point.x, point.y))))
            )

            selected_points = gpd.GeoDataFrame(geometry=[shapely.geometry.Point(xy) for xy in kmeans.cluster_centers_],
                                               crs=points_gdf.crs)
            final_points.append(selected_points)

        final_points_gdf = gpd.GeoDataFrame(pd.concat(final_points, ignore_index=True), crs=points_gdf.crs)

        return final_points_gdf
    """FIM Distribuição Best Sampling"""

    # =================================================
    """Distribuição systematic"""
    def _generate_systematic_sample_points(self):
        print("Atention! systematic distribution ignores sample_number.")
        valid_points = self.create_all_possible_points()

        if len(valid_points) < self.sample_number:
            print("Unable to generate all plots with the established criteria.")

        if valid_points:
            points_gdf = gpd.GeoDataFrame(geometry=valid_points, crs=self.shp.crs)
            return points_gdf
        else:
            print("Unable to generate plots with the established criteria.")
            return None
        pass
    """FIM Distribuição systematic"""
    # =================================================
    """Distribuição systematic custom"""
    def _generate_systematic_custom_sample_points(self):

        if self.x_y_angle is None:
            print(
                "Erro! When using 'systematic custom' you must define x_y_angle=(x distance (m), y distance(m), angle (degree)")
            sys.exit()

        if not (isinstance(self.x_y_angle, tuple) and len(self.x_y_angle) == 3 and
                all(isinstance(x, (int, float)) for x in self.x_y_angle) and
                0 <= self.x_y_angle[2] <= 360):
            print(
                "Erro! When using 'systematic custom' you must define x_y_angle=(x distance (m), "
                "y distance(m), angle (degree)) as a tuple of three numeric values, with the angle between 0 and 360 degrees.")
            sys.exit()
        print("Atention! systematic custom distribution ignores sample_number.")

        x_spacing, y_spacing, rotation_angle = self.x_y_angle

        # Extensão da camada
        extent = self.reduced_shp.total_bounds
        x_min, y_min, x_max, y_max = extent

        if not (np.isfinite(x_min) and np.isfinite(y_min) and np.isfinite(x_max) and np.isfinite(y_max)):
            print("Layer extent contains infinite or NaN values.")
            return None

        if x_min >= x_max or y_min >= y_max:
            print("Invalid range of x or y values.")
            return None

        # Geração dos pontos do grid
        x_coords = np.arange(x_min + x_spacing / 2, x_max, x_spacing)
        y_coords = np.arange(y_min + y_spacing / 2, y_max, y_spacing)

        points = []
        for x in x_coords:
            for y in y_coords:
                points.append(Point(x, y))

        # Criando uma GeoDataFrame a partir dos pontos gerados
        grid_gdf = gpd.GeoDataFrame(geometry=points, crs=self.shp.crs)

        # Calcular o centroide da extensão para ser o ponto de origem da rotação
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        # Rotacionando a grade
        rotated_grid = grid_gdf.copy()
        rotated_grid['geometry'] = [rotate(point, rotation_angle, origin=(center_x, center_y)) for point in
                                    grid_gdf['geometry']]

        # Filtrando os pontos que estão dentro da forma reduzida (reduced_shp)
        valid_points = rotated_grid[rotated_grid.intersects(self.reduced_shp.unary_union)]

        if not valid_points.empty:
            return valid_points
        else:
            print("Unable to generate plots with the established criteria.")
            return None
    # =================================================
    """FIM Distribuição systematic custom"""

    def _create_buffer(self, points):
        if self.plot_format == 'round':
            radius = math.sqrt(self.plot_area / math.pi)
            buffer = points.geometry.buffer(radius)
            buffer_gdf = gpd.GeoDataFrame(geometry=buffer, crs=points.crs)
            return buffer_gdf

        if self.plot_format == 'squared':
            side_length = math.sqrt(self.plot_area)
            half_side = side_length / 2

            squares = []
            for point in points.geometry:
                square = Polygon([
                    (point.x - half_side, point.y - half_side),
                    (point.x + half_side, point.y - half_side),
                    (point.x + half_side, point.y + half_side),
                    (point.x - half_side, point.y + half_side),
                    (point.x - half_side, point.y - half_side)
                ])
                squares.append(square)

            buffer_gdf = gpd.GeoDataFrame(geometry=squares, crs=points.crs)
            return buffer_gdf

        if self.plot_format == 'rectangle':
            half_width = self.rectangle_size[0] / 2
            half_height = self.rectangle_size[1] / 2

            rectangles = []
            for point in points.geometry:
                rectangle = Polygon([
                    (point.x - half_width, point.y - half_height),
                    (point.x + half_width, point.y - half_height),
                    (point.x + half_width, point.y + half_height),
                    (point.x - half_width, point.y + half_height),
                    (point.x - half_width, point.y - half_height)
                ])
                rectangles.append(rectangle)

            buffer_gdf = gpd.GeoDataFrame(geometry=rectangles, crs=points.crs)
            return buffer_gdf

        return None
    def _save_sample_points_to_shp(self, sample_points, filename):
        """Saves the generated sample points to a Shapefile.

        Args:
            sample_points (GeoDataFrame): The GeoDataFrame containing the sample points.
            filename (str): The desired filename for the Shapefile (including extension).

        """
        # Check if sample_points is a GeoDataFrame
        if not isinstance(sample_points, gpd.GeoDataFrame):
            raise TypeError("sample_points must be a GeoDataFrame")

        # Save the GeoDataFrame to a Shapefile
        sample_points.to_file(filename)
        print(f"Sample points saved to Shapefile: {filename}")
    def _save_sample_buffer_to_shp(self, buffer, filename):
        # Check if sample_points is a GeoDataFrame
        if not isinstance(buffer, gpd.GeoDataFrame):
            raise TypeError("buffer must be a GeoDataFrame")

        # Save the GeoDataFrame to a Shapefile
        buffer.to_file(filename)
        print(f"Buffer saved to Shapefile: {filename}")

    def _visualize_points(self, points, buffer=None):
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.axis('off')

        # Plot the polygons
        self.polygons.plot(ax=ax, facecolor='gray', edgecolor='black')

        # Plot the buffer if provided
        if buffer is not None:
            buffer.plot(ax=ax, color='green', alpha=0.5)

        # Plot the points if provided
        if points is not None:
            points.plot(ax=ax, marker='o', color='red', markersize=50)

            # Calculate total sampled area
            total_sampled_area = points.shape[0] * self.plot_area
            total_area = self.polygons.area.sum()
            percentual_area = (total_sampled_area / total_area) * 100

            # Display total sampled area as annotation
            ax.annotate(
                f'Total area: {round(total_area, 2)}m² | Sampled size: {total_sampled_area}m² | '
                f'Percentual sampled: {percentual_area:.2f}% | Total samples: {points.shape[0]} | '
                f'Sample size: {self.plot_area}m²',
                xy=(0.5, 1), xycoords='axes fraction', ha='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black")
            )

        plt.show()