# Generated by Django 5.2.3 on 2025-06-19 13:34

import django.contrib.gis.db.models.fields
import django.contrib.postgres.fields
import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Camera',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('camera_name', models.CharField(max_length=255, unique=True)),
                ('serial_number', models.CharField(blank=True, max_length=100, null=True, unique=True)),
                ('model', models.CharField(blank=True, max_length=100, null=True)),
                ('lens', models.CharField(blank=True, max_length=100, null=True)),
                ('focal_length_mm', models.FloatField(blank=True, null=True)),
                ('sensor_width_mm', models.FloatField(blank=True, null=True)),
                ('sensor_height_mm', models.FloatField(blank=True, null=True)),
                ('pixel_size_um', models.FloatField(blank=True, null=True)),
                ('easting', models.FloatField(blank=True, null=True)),
                ('northing', models.FloatField(blank=True, null=True)),
                ('elevation', models.FloatField(blank=True, null=True)),
                ('epsg_code', models.IntegerField(default=32632)),
                ('installation_date', models.DateField(blank=True, null=True)),
                ('notes', models.TextField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('location', django.contrib.gis.db.models.fields.PointField(blank=True, null=True, srid=32632)),
            ],
        ),
        migrations.CreateModel(
            name='Image',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('acquisition_timestamp', models.DateTimeField()),
                ('file_path', models.CharField(max_length=1024, unique=True)),
                ('image_width_px', models.IntegerField(blank=True, null=True)),
                ('image_height_px', models.IntegerField(blank=True, null=True)),
                ('exif_data', models.JSONField(blank=True, null=True)),
                ('checksum_md5', models.CharField(blank=True, max_length=32, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('camera', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name='images', to='glacier_monitoring_app.camera')),
            ],
        ),
        migrations.CreateModel(
            name='DICAnalysis',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('analysis_timestamp', models.DateTimeField()),
                ('master_image_path', models.CharField(max_length=1024)),
                ('slave_image_path', models.CharField(max_length=1024)),
                ('master_timestamp', models.DateTimeField()),
                ('slave_timestamp', models.DateTimeField()),
                ('software_used', models.CharField(blank=True, max_length=100, null=True)),
                ('software_version', models.CharField(blank=True, max_length=50, null=True)),
                ('processing_parameters', models.JSONField(blank=True, null=True)),
                ('time_difference_hours', models.FloatField(blank=True, null=True)),
                ('notes', models.TextField(blank=True, null=True)),
                ('reference_image', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='reference_analyses', to='glacier_monitoring_app.image')),
                ('secondary_image', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='secondary_analyses', to='glacier_monitoring_app.image')),
            ],
        ),
        migrations.CreateModel(
            name='CameraCalibration',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('calibration_date', models.DateTimeField()),
                ('colmap_model_id', models.IntegerField(choices=[(0, 'SIMPLE_PINHOLE'), (1, 'PINHOLE'), (2, 'SIMPLE_RADIAL'), (3, 'RADIAL'), (4, 'OPENCV'), (5, 'OPENCV_FISHEYE'), (6, 'FULL_OPENCV'), (7, 'FOV'), (8, 'SIMPLE_RADIAL_FISHEYE'), (9, 'RADIAL_FISHEYE'), (10, 'THIN_PRISM_FISHEYE')])),
                ('colmap_model_name', models.CharField(blank=True, max_length=50, null=True)),
                ('image_width_px', models.IntegerField()),
                ('image_height_px', models.IntegerField()),
                ('intrinsic_params', django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None)),
                ('rotation_quaternion', django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), blank=True, null=True, size=4)),
                ('translation_vector', django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), blank=True, null=True, size=3)),
                ('is_active', models.BooleanField(default=True)),
                ('notes', models.TextField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('camera', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='calibrations', to='glacier_monitoring_app.camera')),
            ],
            options={
                'constraints': [models.UniqueConstraint(condition=models.Q(('is_active', True)), fields=('camera', 'is_active'), name='unique_active_calibration')],
            },
        ),
        migrations.CreateModel(
            name='DICResult',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('seed_x_ref_px', models.IntegerField()),
                ('seed_y_ref_px', models.IntegerField()),
                ('target_x_sec_px', models.FloatField(blank=True, null=True)),
                ('target_y_sec_px', models.FloatField(blank=True, null=True)),
                ('displacement_x_px', models.FloatField(blank=True, null=True)),
                ('displacement_y_px', models.FloatField(blank=True, null=True)),
                ('correlation_score', models.FloatField(blank=True, null=True)),
                ('status_flag', models.CharField(blank=True, max_length=50, null=True)),
                ('analysis', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='results', to='glacier_monitoring_app.dicanalysis')),
            ],
            options={
                'indexes': [models.Index(fields=['analysis'], name='glacier_mon_analysi_c89806_idx')],
                'constraints': [models.UniqueConstraint(fields=('analysis', 'seed_x_ref_px', 'seed_y_ref_px'), name='unique_seed_point')],
            },
        ),
        migrations.AddIndex(
            model_name='image',
            index=models.Index(fields=['camera'], name='glacier_mon_camera__99efcf_idx'),
        ),
        migrations.AddIndex(
            model_name='image',
            index=models.Index(fields=['acquisition_timestamp'], name='glacier_mon_acquisi_b2e295_idx'),
        ),
        migrations.AddConstraint(
            model_name='image',
            constraint=models.UniqueConstraint(fields=('camera', 'acquisition_timestamp'), name='unique_camera_timestamp'),
        ),
        migrations.AddIndex(
            model_name='dicanalysis',
            index=models.Index(fields=['master_image_path'], name='glacier_mon_master__fdd783_idx'),
        ),
        migrations.AddIndex(
            model_name='dicanalysis',
            index=models.Index(fields=['slave_image_path'], name='glacier_mon_slave_i_024d69_idx'),
        ),
        migrations.AddIndex(
            model_name='dicanalysis',
            index=models.Index(fields=['master_timestamp'], name='glacier_mon_master__12357d_idx'),
        ),
        migrations.AddIndex(
            model_name='dicanalysis',
            index=models.Index(fields=['slave_timestamp'], name='glacier_mon_slave_t_daeb35_idx'),
        ),
        migrations.AddConstraint(
            model_name='dicanalysis',
            constraint=models.CheckConstraint(condition=models.Q(('master_image_path', models.F('slave_image_path')), _negated=True), name='different_image_paths'),
        ),
        migrations.AddConstraint(
            model_name='dicanalysis',
            constraint=models.UniqueConstraint(fields=('master_image_path', 'slave_image_path'), name='unique_image_pair_paths'),
        ),
    ]
